#=
    SCALE-TRACK

    SCALE-TRACK is advancing two-way coupled Euler-Lagrange particle tracking
    to the realm of exascale computing enabling the simulation of dispersed
    multiphase flows at lower cost and energy consumption.  This is achieved
    through a coupling algorithm, which eliminates synchronisation barriers,
    and new cache-friendly data structures.

    Copyright (C) 2024 Sergey Lesnik
    Copyright (C) 2024 Henrik Rusche

    This file is part of SCALE-TRACK.

    SCALE-TRACK is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along
    with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.
=#

tNow = time()
const reg = Dict()
reg["gc_num"] = Base.gc_num()

using Random
# using Distributions  # Hangs when profiling with Nsight
using WriteVTK
using CUDA
using StaticArrays
using BenchmarkTools
import Adapt
import Base: *, Event
using Accessors
using MPI
using Base.Threads

ΔtLoadModules = time() - tNow

###############################################################################
# Compile-time globals

# Debug communication logging.  If true, Log to stdout from every rank prefixed
# by [rank].
DebugComm = false

###############################################################################
# Auxiliary Methods

function timing(t, s)
    dt = round(time() - t, sigdigits=4)
    println(s, " in ", dt, " s")

    # Invoke flush to ensure immediate printing even when executed within C code
    flush(stdout)
    return time()
end

# Macro for printing debug statements.  Enable by setting global DebugComm to
# true.  If disabled, all debug printing is turned off with zero overhead.
macro debugCommPrintln(ex)
    if DebugComm
        # Put message into a single string before printing to avoid output
        # overlap
        msg = :(Main.Base.inferencebarrier(Main.Base.string)(
            "[", comm.rank, "] ", $(esc(ex)), "\n"
        ))
        return :( print($msg) )
    end
    return nothing
end

# Wrapper for MPI non-blocking synchronous send unavailable in MPI.jl
function comm_Issend(buf::MPI.Buffer, dest::Integer, tag::Integer, comm::MPI.Comm,
        req::MPI.AbstractRequest=MPI.Request()
)
    @assert MPI.isnull(req)
    # int MPI_Issend(const void* buf, int count, MPI_Datatype datatype, int
    #               dest, int tag, MPI_Comm comm, MPI_Request *request)
    MPI.API.MPI_Issend(buf.data, buf.count, buf.datatype, dest, tag, comm, req)
    MPI.setbuffer!(req, buf)
    return req
end

# Wrapper for MPI Iprobe that does not allocate flag
function comm_Iprobe(
    comm::MPI.Comm, flag, status=nothing;
    source::Integer=MPI.API.MPI_ANY_SOURCE[], tag::Integer=MPI.API.MPI_ANY_TAG[]
)
    MPI.API.MPI_Iprobe(
        source, tag, comm, flag, something(status, MPI.API.MPI_STATUS_IGNORE[])
    )
    return flag[] != 0
end

# Wrapper for MPI Test that does not allocate flag
function comm_test(
    req::MPI.AbstractRequest, flag,
    status::Union{Ref{MPI.Status}, Nothing}=nothing
)
    # int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
    MPI.API.MPI_Test(req, flag, something(status, MPI.API.MPI_STATUS_IGNORE[]))
    if MPI.isnull(req)
        MPI.setbuffer!(req, nothing)
    end
    return flag[] != 0
end

# Alternative implementation of Base.wait that allocates flag only once
function comm_wait(req::MPI.Request)
    flag = Ref{Cint}()
    while !comm_test(req, flag)
        yield()
    end
end

###############################################################################
# Type alias

const scalar = Float32
const label = Int32

# A shorthand notation for custom types and multiplication operation for easier
# introduction of the type primitives into the code
struct LBL end
(*)(n, ::Type{LBL}) = label(n)
struct SCL end
(*)(n, ::Type{SCL}) = scalar(n)


###############################################################################
# Structs and their constructors

struct LabelVec <: FieldVector{3, label}
    x::label
    y::label
    z::label
end

struct ScalarVec <: FieldVector{3, scalar}
    x::scalar
    y::scalar
    z::scalar
end

mutable struct MutScalarVec <: FieldVector{3, scalar}
    x::scalar
    y::scalar
    z::scalar
end

const ScalarField = Vector{scalar}      # @SiSc: added scalar field
Adapt.@adapt_structure ScalarField

const VectorField = Vector{ScalarVec}
Adapt.@adapt_structure VectorField

# @SiSc: added additional fields
struct TwoWayEulerian{TV, TS}
    N::label
    U::TV
    T::TS
    rhoV::TS
    UTrans::TV
    hTrans::TS
    rhoVTrans::TS
end

# @SiSc: added additional fields
function TwoWayEulerian{TV, TS}(N) where {TV, TS}
    TwoWayEulerian{TV, TS}(
        N,
        TV(undef, N),
        TS(undef, N),
        TS(undef, N),
        TV(undef, N),
        TS(undef, N),
        TS(undef, N)
    )
end
Adapt.@adapt_structure TwoWayEulerian

# Mesh is limited to a cuboid defined by two points origin and ending with L
# containing length in each direction and N number of cells per direction
struct Mesh
    N::LabelVec
    origin::ScalarVec
    ending::ScalarVec
    L::ScalarVec
    Δ::ScalarVec
    rΔ::ScalarVec
    decomposition::LabelVec
    decompositionXTimesY::label
    partitionN::LabelVec
    partitionNxTimesNy::label
end

function construct_mesh(N, origin, ending, decomposition)
    L = ending .- origin
    Δ = L ./ N
    rΔ = 1.0SCL ./ Δ
    if (sum(rem.(N, decomposition)) != 0)
        throw(
            ErrorException(
                string(
                    "Each partition must have the same number of cells per",
                    " direction.  The current setup does not comply:\n",
                    " N = ", N, " decomposition = ", decomposition
                )
            )
        )
    end
    decompositionXTimesY = decomposition[1]*decomposition[2]
    partitionN = N ./ decomposition
    partitionNxTimesNy = partitionN[1]*partitionN[2]
    Mesh(
        N, origin, ending, L, Δ, rΔ, decomposition, decompositionXTimesY,
        partitionN, partitionNxTimesNy
    )
end

function Mesh(
    NInt::Integer, originR::Real, endingR::Real, decomposition=(1,1,1)
)
    N = [NInt, NInt, NInt]
    origin = [originR, originR, originR]
    ending = [endingR, endingR, endingR]
    construct_mesh(N, origin, ending, decomposition)
end

Adapt.@adapt_structure Mesh

struct Time
    t::scalar
    Δt::scalar
end

# Locks and events that help to orchestrate the async tasks and threads
struct Locks
    chunkTransfers::ReentrantLock
    eulerianComms::Vector{ReentrantLock}
    eulerianRequest::ReentrantLock
end

function Locks(nEulerian)
    eulerianComms = fill(ReentrantLock(), nEulerian)
    Locks(ReentrantLock(), eulerianComms, ReentrantLock())
end

struct Events
    U_copied::Event
    S_copied::Event
    Eulerian_computed::Event

    Events() = new(Event(true), Event(true), Event(true))
end

# Extrapolation of sources
abstract type AbstractExtrapolator end
struct ConstExtrapolator <: AbstractExtrapolator
    prevUTrans::VectorField
    prevHTrans::ScalarField
    prevRhoVTrans::ScalarField
end

function ConstExtrapolator(N::Real)
    uf = VectorField(undef, N) # velocity
    hf = ScalarField(undef, N) # energy/heat
    rf = ScalarField(undef, N) # vapour density
    fill!(uf, ScalarVec(0, 0, 0))
    fill!(hf, 0SCL)
    fill!(rf, 0SCL)
    ConstExtrapolator(uf, hf, rf)
end

struct Control{E <: AbstractExtrapolator}
    locks::Locks
    events::Events
    extrapolator::E
end

struct BoundingBox{T}
    min::T
    max::T
end

function BoundingBox{T}() where {T}
    BoundingBox{T}(T(undef, 3), T(undef, 3))
end

Adapt.@adapt_structure BoundingBox

# @SiSc: added additional and modified some properties
# Superscripts: c - carrier phase; d - dispersed phase
struct Chunk{T, A}
    N::label
    μᶜ::scalar      # dynamic viscosity (continuous phase)
    ρᶜ::scalar      # density (continuous phase)
    ρᵈ::scalar      # density (disperse phase)
    gX::scalar      # gravitational acceleration
    gY::scalar      # gravitational acceleration
    gZ::scalar      # gravitational acceleration
    Cₚᶜ::scalar     # heat capacity (carrier phase)
    Cₚᵈ::scalar     # heat capacity (dispersed phase)
    Dᵈᶜ::scalar     # diffusion coefficient of water vapour in air
    Mᵈ::scalar      # molar weigth of water
    σᶜ::scalar      # surface tension of water in air
    RG::scalar      # gas constant
    SLH::scalar     # specific latent heat of water vaporisation
    boundingBox::BoundingBox{T}
    time::A
    d::T            # diameter
    X::T            # position, x-component
    Y::T            # position, y-component
    Z::T            # position, z-component
    U::T            # velocity, x-component
    V::T            # velocity, y-component
    W::T            # velocity, z-component
    T::T            # temperature
end

# @SiSc: added additional and modified some properties
function Chunk{T, A}(
    N, μᶜ, ρᶜ, ρᵈ, gX, gY, gZ, Cₚᶜ, Cₚᵈ, Dᵈᶜ, Mᵈ, σᶜ, RG, SLH
) where {T, A}
    Chunk{T, A}(
        N,
        μᶜ,
        ρᶜ,
        ρᵈ,
        gX,
        gY,
        gZ,
        Cₚᶜ,
        Cₚᵈ,
        Dᵈᶜ,
        Mᵈ,
        σᶜ,
        RG,
        SLH,
        BoundingBox{T}(),
        A(undef, 1),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N)
    )
end

Adapt.@adapt_structure Chunk

# Communication types based on MPI
abstract type CommMember end

struct Master <: CommMember
    deviceNumber::label
    requiredEulerianRanks::Set{label}
    inquiringEulerianRanks::Vector{label}
end

Master(deviceNumber) = Master(deviceNumber, Set{label}(), Vector{label}())

struct Slave <: CommMember
    inquiringEulerianRanks::Vector{label}
end

Slave() = Slave(Vector{label}())

struct Comm{T<:CommMember}
    member::T
    communicator::MPI.Comm
    isMaster::Bool
    isHost::Bool
    rank::Integer
    jlRank::Integer
    masterRank::Integer
    size::Integer
end

function Comm(member, communicator)
    if MPI.Initialized()
        rank = MPI.Comm_rank(communicator)
        isMaster = (rank == 0)
        isHost = (typeof(member) == Master)
        return Comm(
            member,
            communicator,
            isMaster,
            isHost,
            rank,
            rank + 1,
            0,
            MPI.Comm_size(communicator)
        )
    else
        rank = 0
        return Comm(member, communicator, true, true, rank, rank + 1, 0, 1)
    end
end

# Structs used for function tagging to identify on which backend the code is
# executed
struct CPU end
struct GPU end

###############################################################################
# Methods
function initComm(executor)
    if MPI.Initialized() && MPI.JULIA_TYPE_PTR_ATTR[]==0
        MPI.run_init_hooks()
    else
        # Construct communication with a single rank and executor
        MPI.Init()
    end

    nDevicesPerNode = count_devices_per_node(executor)
    shmComm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, 0)
    shmRank = MPI.Comm_rank(shmComm)
    nRanksPerNode = MPI.Allreduce(shmRank, max, shmComm) + 1
    hostStride = nRanksPerNode ÷ nDevicesPerNode
    # Prevent 0 stride
    hostStride = hostStride < 1 ? 1 : hostStride

    # Assign devices to ranks uniformly
    hostRanks =
        range(0, step=hostStride, length=min(nRanksPerNode, nDevicesPerNode))
    if shmRank in hostRanks
        deviceNumber = shmRank ÷ hostStride
        CUDA.device!(deviceNumber)
        comm = Comm(Master(deviceNumber), MPI.COMM_WORLD)
    else
        comm = Comm(Slave(), MPI.COMM_WORLD)
    end

    if !DebugComm && !comm.isMaster
        # Suppress output from non-master ranks
        redirect_stdout(devnull)
    end

    return comm
end

function count_devices_per_node(::GPU)
    return length(CUDA.devices())
end

function allocate_chunk(::CPU, constructorArgs...)
    return Chunk{Vector{scalar}, Vector{Time}}(constructorArgs...)
end

function allocate_chunk(::GPU, constructorArgs...)
    return Chunk{CuVector{scalar}, CuVector{Time}}(constructorArgs...)
end

function allocate_chunk(::Comm{Master}, executor, constructorArgs...)
    return allocate_chunk(executor, constructorArgs...)
end

function set_time!(chunk, t, Δt, ::CPU)
    chunk.time[1] = Time(t, Δt)
end

function set_time!(chunk, t, Δt, ::GPU)
    CUDA.@allowscalar chunk.time[1] = Time(t, Δt)
end

function increment_time!(chunk, Δt, ::Comm{Master}, executor)
    increment_time!(chunk, Δt, executor)
end

function increment_time!(chunk, Δt, executor::CPU)
    set_time!(chunk, chunk.time[1].t + Δt, Δt, executor)
end

function increment_time!(chunk, Δt, executor::GPU)
    CUDA.@allowscalar t = chunk.time[1].t
    set_time!(chunk, t + Δt, Δt, executor)
end

function default_rng(::CPU)
    return Random.default_rng()
end

# Use CUDA random number generator to enable processing on the device without
# copying to host
function default_rng(::GPU)
    return CUDA.default_rng()
end

function init!(chunk, mesh, executor, randSeed=19891)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)

    rng = default_rng(executor)
    Random.seed!(rng, randSeed)
    fill!(c.boundingBox.min, 0.0)
    fill!(c.boundingBox.max, 0.0)
    rand!(rng, c.d)
    rand!(rng, c.X)
    rand!(rng, c.Y)
    rand!(rng, c.Z)
    @. c.X = c.X*mesh.L.x + mesh.origin.x
    @. c.Y = c.Y*mesh.L.y + mesh.origin.y
    @. c.Z = c.Z*mesh.L.z + mesh.origin.z
    @. c.d = c.d*3e-6SCL + 2e-6SCL
    fill!(c.T, 288.15SCL)   # @SiSc: added temperature
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    return nothing
end

function init_random!(field, interval, offset)
    rng = Random.default_rng()
    Random.seed!(rng, 19891)
    rand!(rng, field)
    map!(x -> x*interval .+ offset, field, field)
    return nothing
end

# Helper to copy all struct data from host to device
function copy!(a, b)
    for n in fieldnames(typeof(a))
        if !(typeof(getfield(b, n)) <: Number)
            typeoffield = typeof(getfield(b, n))
            if (typeof(getfield(b, n)) <: AbstractArray)
                copyto!(getfield(a, n), getfield(b, n))
            else
                copy!(getfield(a, n), getfield(b, n))
            end
        end
    end
    return nothing
end

# Compute cell and partition indices along each direction.  Indices start at 0.
@inline function locate_ijk(x, y, z, mesh)
    # TODO unsafe_trunc instead of floor is 20% faster but then numbers within
    # -1 < n < 0 are truncated to 0 which leads to incorrect localization of
    # the parcel
    iGlobal = floor(label, (x - mesh.origin.x)*mesh.rΔ.x)
    jGlobal = floor(label, (y - mesh.origin.y)*mesh.rΔ.y)
    kGlobal = floor(label, (z - mesh.origin.z)*mesh.rΔ.z)
    iPartition, iLocal = divrem(iGlobal, mesh.partitionN.x)
    jPartition, jLocal = divrem(jGlobal, mesh.partitionN.y)
    kPartition, kLocal = divrem(kGlobal, mesh.partitionN.z)
    return (iGlobal, jGlobal, kGlobal, iLocal, jLocal, kLocal, iPartition,
        jPartition, kPartition
    )
end

# Compute linear cell index, the corresponding indices along each direction and
# linear partition index.  Linear indices start at 1 and direction indices at 0.
@inline function locate(x, y, z, mesh)
    iGlobal, jGlobal, kGlobal, iLocal, jLocal, kLocal, iPartition, jPartition,
        kPartition = locate_ijk(x, y, z, mesh)
    partitionI = (
        abs(kPartition)*mesh.decompositionXTimesY
        + abs(jPartition)*mesh.decomposition.x + abs(iPartition) + 1LBL
    )
    posI = (
        abs(kLocal)*mesh.partitionNxTimesNy
        + abs(jLocal)*mesh.partitionN.x + abs(iLocal) + 1LBL
    )
    return (iGlobal, jGlobal, kGlobal, posI, partitionI)
end

@inline function locate(pos, mesh)
    return locate(pos.x, pos.y, pos.z, mesh)
end

# @SiSc: set_parcel_state removed
# @SiSc: included thermal energy and mass transfer, removed state0
@inline function update!(
    eulerian::TwoWayEulerian, dUTrans, dhTrans, drhoVTrans, posI, ::CPU
)
    # momentum transfer
    @inbounds eulerian.UTrans[posI] += dUTrans

    # thermal energy transfer
    @inbounds eulerian.hTrans[posI] += dhTrans

    # mass transfer
    @inbounds eulerian.rhoVTrans[posI] += drhoVTrans
    return nothing
end

# @SiSc:
# included additional fields, removed state0
@inline function update!(
    eulerian::TwoWayEulerian, dUTrans, dhTrans, drhoVTrans, posI, ::GPU
)
    @inbounds begin
        # momentum transfer
        for i=1LBL:3LBL
            # Reinterpret the eulerian field at a location specified by posI
            # with an offset as scalar and get pointer to it.
            scalar_ptr = pointer(
                reinterpret(scalar, eulerian.UTrans), (posI-1LBL)*3LBL+i
            )
            GC.@preserve CUDA.atomic_add!(scalar_ptr, dUTrans[i])
        end

        # thermal energy transfer
        CUDA.@atomic eulerian.hTrans[posI] += dhTrans

        # mass transfer
        CUDA.@atomic eulerian.rhoVTrans[posI] += drhoVTrans
    end
    return nothing
end


# @SiSc: removed update at boundary

# Constant extrapolator assumes that the current source is the same as the true
# from the previous time step.  The estimated source from the previous time
# step is corrected using the true source from the previous time step.
# estSⁿ = trueSⁿ⁻¹ - estSⁿ⁻¹ + extrapSⁿ, with extrapSⁿ = trueSⁿ⁻¹
# @SiSc: added fields for h and rhoV
function estimate_source!(
    currUTrans, currHTrans, currRhoVTrans, extrapolator::ConstExtrapolator
)
#    currUTrans .= 2.0.*currUTrans .- extrapolator.prevUTrans
#    for i in eachindex(control.extrapolator.prevUTrans)
#        @inbounds extrapolator.prevUTrans[i] = currUTrans[i]
#    end

#    currHTrans .= 2.0*currHTrans .- extrapolator.prevHTrans
#    for i in eachindex(control.extrapolator.prevHTrans)
#        @inbounds extrapolator.prevHTrans[i] = currHTrans[i]
#    end

#    currRhoVTrans .= 2.0*currRhoVTrans .- extrapolator.prevRhoVTrans
#    for i in eachindex(control.extrapolator.prevRhoVTrans)
#        @inbounds extrapolator.prevRhoVTrans[i] = currRhoVTrans[i]
#    end
end

# Find an extremum by comparing the element i of array arr with the value val
# using operation op
@inline function atomic_extremum!(arr, i, val, op)
    ptr = pointer(arr, i)
    old = arr[i]
    while true
        assumed = old
        old = CUDA.atomic_cas!(ptr, assumed, op(arr[i], val))
        (assumed != old) || break  # mimic do-while loop
    end
    return nothing
end

# Calculation of the axis aligned bounding box mainly consists of identifying
# global minima and maxima of particles' positions.  This is done in two
# stages.  First, each GPU block finds the extrema of particles contained
# within the block by using atomics and shared memory.  Second, the bounding
# box of the chunk is computed using atomics and global memory.
@inline function update!(boundingBox, pos, ::GPU)
    @inbounds begin
        # Allocate and initialize shared memory for min/max per block
        s = CUDA.CuStaticSharedArray(scalar, 6)
        sMin = view(s, 1:3)
        sMax = view(s, 4:6)
        if threadIdx().x == 1
            for i in eachindex(sMin)
                sMin[i] = scalar(Inf)
                sMax[i] = scalar(-Inf)
            end
        end
        sync_threads()

        # Calculate min/max per block using atomics on shared memory
        for i in eachindex(sMin)
            atomic_extremum!(sMin, i, pos[i], Base.min)
            atomic_extremum!(sMax, i, pos[i], Base.max)
        end
        sync_threads()

        # Calculate global min/max using atomics on global memory
        if threadIdx().x == 1
            for i in eachindex(boundingBox.min)
                atomic_extremum!(boundingBox.min, i, sMin[i], Base.min)
                atomic_extremum!(boundingBox.max, i, sMax[i], Base.max)
            end
        end
    end
    return nothing
end

# Identify partition linear indices (i.e. ranks) belonging to the bounding box
function determine!(requiredEulerianRanks, chunkBoundingBox, mesh, control)
    _, _, _, _, _, _, minP... = locate_ijk(chunkBoundingBox.min..., mesh)
    _, _, _, _, _, _, maxP... = locate_ijk(chunkBoundingBox.max..., mesh)
    # Promote to one-based indexing scheme
    minP = minP .+ 1
    maxP = maxP .+ 1
    d = mesh.decomposition
    li = LinearIndices((1:d.x, 1:d.y, 1:d.z))
    lock(control.locks.eulerianRequest) do
        union!(
            requiredEulerianRanks,
            Set{label}(
                li[minP[1]:maxP[1], minP[2]:maxP[2], minP[3]:maxP[3]]
                .- 1  # Rank indexing is zero-based
            )
        )
    end
end

# Evolve particles using CPU
function evolve!(chunk, eulerian, mesh, Δt, control, executor::CPU)
    @inbounds begin
        nSteps = 10LBL
        ΔtP = chunk.time[1].Δt / nSteps
        for i = 10LBL:chunk.N
            evolve_particle!(chunk, eulerian, i, ΔtP, mesh, nSteps, executor)
        end

        tEvolve = time() - tStart
        if !firstPass
            global totalTime += tEvolve
        end
        global firstPass = false
        println("Lagrangian solver timings: current evolve = ",
            round(tEvolve, sigdigits=4), " s; total time = ",
            round(totalTime, sigdigits=4), " s"
        )
    end
    return nothing
end

function serve_eulerian(inquiringEulerianRanks, eulerian, comm, control)
    # @SiSc: changed to sreqsU; added T, rhoV
    sreqsU = Vector{MPI.Request}(undef, length(inquiringEulerianRanks))
    sreqsT = Vector{MPI.Request}(undef, length(inquiringEulerianRanks))
    sreqsrhoV = Vector{MPI.Request}(undef, length(inquiringEulerianRanks))
    for (i, inqHost) in enumerate(inquiringEulerianRanks)
        lock(control.locks.eulerianComms[comm.jlRank]) do
            @debugCommPrintln("Send U to $inqHost")
            # @SiSc: changed to sreqsU
            sreqsU[i] = MPI.Isend(
                eulerian[comm.jlRank].U, comm.communicator, dest=inqHost,
                tag=0
            )
            # @SiSc: added T, rhoV
            # questions @Sergey:
            #  is this implementatin correct?
            #  do we need additional sreqs for T and rhoV?
            @debugCommPrintln("Send T to $inqHost")
            sreqsT[i] = MPI.Isend(
                eulerian[comm.jlRank].T, comm.communicator, dest=inqHost,
                tag=0
            )
            @debugCommPrintln("Send rhoV to $inqHost")
            sreqsrhoV[i] = MPI.Isend(
                eulerian[comm.jlRank].rhoV, comm.communicator, dest=inqHost,
                tag=0
            )
        end
    end
    return (sreqsU, sreqsT, sreqsrhoV) # @SiSc: changed to sreqsU; added T, rhoV
end

function receive_sources(inquiringEulerianRanks, comm, eulerian)
    inqRanks = inquiringEulerianRanks
    # @SiSc: added T/h, rhoV
    # questions @Sergey:
    #  is this implementatin correct?
    #  do we need additional sourceRreqs and sourceBuffers for T and rhoV?
    #  is it correct to have them as ScalarField?
    sourceRreqsU = Vector{MPI.Request}(undef, length(inqRanks))
    sourceRreqsT = Vector{MPI.Request}(undef, length(inqRanks))
    sourceRreqsrhoV = Vector{MPI.Request}(undef, length(inqRanks))
    sourceBuffersU = Vector{VectorField}(undef, length(inqRanks))
    sourceBuffersT = Vector{ScalarField}(undef, length(inqRanks))
    sourceBuffersrhoV = Vector{ScalarField}(undef, length(inqRanks))
    if !isempty(inqRanks)
        for (i, inqRank) in enumerate(inqRanks)
            @debugCommPrintln("Receive U source from $inqRank")
            sourceBuffersU[i] = similar(eulerian[comm.jlRank].UTrans)
            sourceRreqsU[i] = MPI.Irecv!(
                sourceBuffersU[i], comm.communicator; source=inqRank, tag=1
            )
            @debugCommPrintln("Receive T source from $inqRank")
            sourceBuffersT[i] = similar(eulerian[comm.jlRank].hTrans)
            sourceRreqsT[i] = MPI.Irecv!(
                sourceBuffersT[i], comm.communicator; source=inqRank, tag=1
            )
            @debugCommPrintln("Receive rhoV source from $inqRank")
            sourceBuffersrhoV[i] = similar(eulerian[comm.jlRank].rhoVTrans)
            sourceRreqsrhoV[i] = MPI.Irecv!(
                sourceBuffersrhoV[i], comm.communicator; source=inqRank, tag=1
            )
        end
    end
    return (
        sourceRreqsU, sourceRreqsT, sourceRreqsrhoV,
        sourceBuffersU, sourceBuffersT, sourceBuffersrhoV
    )
end

function init_async_evolve!(eulerian, control, comm::Comm{Slave}, ::GPU)
    # The infinite loop to be run inside an asynchronous task that is
    # specifically yielded at "lock" and "wait"
    GC.enable(false)
    while true
        # Non-blocking consensus for processing Eulerian requests
        barrierFlag = Ref{Cint}(0)
        probeFlag = Ref{Cint}(0)
        # Nothing to send, hence, set the barrier directly
        breq = MPI.Ibarrier(comm.communicator)
        sreqs = Vector{MPI.Request}()
        inqRanks = comm.member.inquiringEulerianRanks
        while barrierFlag[] == 0
            comm_Iprobe(comm.communicator, probeFlag; tag=2)
            if probeFlag[] != 0
                probeFlag[] = 0
                bufRef = Ref(41)
                _, status = MPI.Recv!(
                    MPI.Buffer(bufRef), comm.communicator, MPI.Status; tag=2
                )
                sreq = MPI.Isend(
                    MPI.Buffer_send(42), status.source, 3, comm.communicator
                )
                push!(sreqs, sreq)

                @debugCommPrintln("Got Eulerian request from $(status.source)")
                lock(control.locks.eulerianRequest) do
                    push!(inqRanks, status.source)
                end
            end
            MPI.Testall(sreqs)
            comm_test(breq, barrierFlag)
        end

        # @SiSc: changed to eulerianSreqsU; added T and rhoV
        for req in sreqs comm_wait(req) end
        eulerianSreqsU, eulerianSreqsT, eulerianSreqsrhoV =
            serve_eulerian(inqRanks, eulerian, comm, control)

        # @SiSc: changed to eulerianSreqsU; added T and rhoV
        # question @Sergey: are additional waits for T and rhoV necessary?
        for req in eulerianSreqsU comm_wait(req) end
        for req in eulerianSreqsT comm_wait(req) end
        for req in eulerianSreqsrhoV comm_wait(req) end

        # Enable garbage collector at the same time as other thread
        GC.enable(true)

        notify(control.events.U_copied)
        wait(control.events.Eulerian_computed)
        GC.enable(false)

        # @SiSc: changed to sourceRreqsU; added T and rhoV
        sourceRreqsU, sourceRreqsT, sourceRreqsrhoV,
            sourceBuffersU, sourceBuffersT, sourceBuffersrhoV =
            receive_sources(inqRanks, comm, eulerian)
        lock(control.locks.eulerianComms[comm.jlRank]) do
            # @SiSc: changed to sourceBuffersU; added T and rhoV
            for (i, req) in enumerate(sourceRreqsU)
                comm_wait(req)
                eulerian[comm.jlRank].UTrans .+= sourceBuffersU[i]
            end
            for (i, req) in enumerate(sourceRreqsT)
                comm_wait(req)
                eulerian[comm.jlRank].hTrans .+= sourceBuffersT[i]
            end
            for (i, req) in enumerate(sourceRreqsrhoV)
                comm_wait(req)
                eulerian[comm.jlRank].rhoVTrans .+= sourceBuffersrhoV[i]
            end
        end

        # @SiSc: added T and rhoV
        estimate_source!(
            eulerian[comm.jlRank].UTrans,
            eulerian[comm.jlRank].hTrans,
            eulerian[comm.jlRank].rhoVTrans,
            control.extrapolator
        )
        empty!(inqRanks)

        notify(control.events.S_copied)
    end
    return nothing
end

# CUDA call with a setup that efficiently maps to the specific GPU
function init_async_evolve!(
    chunks, eulerian, mesh, control, comm::Comm{Master}, executor::GPU
)
    GC.enable(false)
    CUDA.device!(comm.member.deviceNumber)
    @debugCommPrintln("Hosting device $(CUDA.device())")

    if !haskey(reg, "deviceEulerian")
        # CUDA containers that own the Eulerian fields on the device
        reg["deviceEulerian"] =
            Vector{TwoWayEulerian{CuVector{ScalarVec},CuVector{scalar}}}(
                undef, comm.size
            ) # @SiSc: added ", CuVector{scalar}"
        # CUDA container for the pointers (CuDeviceVector) to the Eulerian
        # containers.  It needs to be stored separately since
        # CuVector{CuVector} is not possible.
        reg["deviceEulerianPointer"] =
            CuVector{TwoWayEulerian{CuDeviceVector{ScalarVec, 1},CuDeviceVector{scalar,1}}}(
                undef, comm.size
            )# @SiSc: added ", CuDeviceVector{scalar,1}"
        for i in eachindex(reg["deviceEulerian"])
            # Initialize Eulerian fields
            reg["deviceEulerian"][i] =
                TwoWayEulerian{CuVector{ScalarVec}, CuVector{scalar}}(
                    eulerian[i].N
                ) # @SiSc: added ", CuVector{scalar}"
            # Get the pointers on the device (cudaconvert) and store them in
            # the pointer container.
            CUDA.@allowscalar reg["deviceEulerianPointer"][i] =
                cudaconvert(reg["deviceEulerian"][i])
        end
    end

    # Compile kernels and prepare configuration
    # Pick any chunk for compilation - only types are important
    aChunk = first(chunks)
    kernel = @cuda launch=false evolve_on_device!(
        aChunk, reg["deviceEulerianPointer"], mesh, executor
    )
    config = launch_configuration(kernel.fun)
    threads = min(aChunk.N, config.threads)
    blocks = cld(aChunk.N, threads)
    bbKernel = @cuda launch=false compute_bounding_box(aChunk, executor)

    # Initialization: bounding boxes of chunks are required before the main
    # kernel may be run
    # Reset the bounding box in device memory
    lock(control.locks.chunkTransfers) do
        @sync begin
            for (i, chunk) in enumerate(chunks)
                @async begin
                    hostChunkBb = BoundingBox{Vector{scalar}}(
                        fill(Inf, 3), fill(-Inf, 3)
                    )
                    copy!(chunk.boundingBox, hostChunkBb)
                    bbKernel(chunk, executor; threads, blocks)
                    copy!(hostChunkBb, chunk.boundingBox)
                    determine!(
                        comm.member.requiredEulerianRanks,
                        hostChunkBb,
                        mesh,
                        control
                    )
                end
            end
        end
    end

    # The infinite loop to be run inside an asynchronous task that is
    # specifically yielded at "lock" and "wait"
    while true
        # Non-blocking consensus for processing Eulerian requests
        reqRanks = lock(control.locks.eulerianRequest) do
            # Exclude self from sending Eulerian requests
            collect(setdiff(comm.member.requiredEulerianRanks, comm.rank))
        end
        nRequests = length(reqRanks)
        # Rank indices are zero-based.  Derive a one-based iterator to iterate
        # over arrays.  Zip these and consecutive range iterators for simpler
        # handling.
        reqRanks₁ = Iterators.map(x -> x+1, reqRanks)
        zipIter = zip(1:nRequests, reqRanks, reqRanks₁)
        sreqs = Vector{MPI.Request}(undef, 2*nRequests)
        # @SiSc: changed to rreqsU; added T and rhoV
        # question @Sergey: necessary and correct?
        rreqsU = Vector{MPI.Request}(undef, nRequests)
        rreqsT = Vector{MPI.Request}(undef, nRequests)
        rreqsrhoV = Vector{MPI.Request}(undef, nRequests)

        # Buffer is empty since only the source of the message is relevant to
        # the receiver
        inqRanks = comm.member.inquiringEulerianRanks
        for (i, iRank₀, iRank₁) in zipIter
            @debugCommPrintln("Request Eulerian from $iRank₀")
            sreqs[i] =
                MPI.Isend(MPI.Buffer_send(42), iRank₀, 2, comm.communicator)
            bufRef = Ref(41)
            sreqs[nRequests + i] =
                MPI.Irecv!(MPI.Buffer(bufRef), iRank₀, 3, comm.communicator)
            # Setup receive for the requested field
            lock(control.locks.eulerianComms[iRank₁])
            # @SiSc: changed to rreqsU; added T and rhoV
            # question @Sergey: necessary and correct?
            rreqsU[i] = MPI.Irecv!(
                eulerian[iRank₁].U, comm.communicator, source=iRank₀, tag=0
            )
            rreqsT[i] = MPI.Irecv!(
                eulerian[iRank₁].T, comm.communicator, source=iRank₀, tag=0
            )
            rreqsrhoV[i] = MPI.Irecv!(
                eulerian[iRank₁].rhoV, comm.communicator, source=iRank₀, tag=0
            )
        end
        barrierOn = false
        barrierFlag = Ref{Cint}(0)
        probeFlag = Ref{Cint}(0)
        breq = MPI.Request()
        while barrierFlag[] == 0
            comm_Iprobe(comm.communicator, probeFlag; tag=2)
            if probeFlag[] != 0
                probeFlag[] = 0
                bufRef = Ref(41)
                _, status = MPI.Recv!(
                    MPI.Buffer(bufRef), comm.communicator, MPI.Status; tag=2
                )
                sreq = MPI.Isend(
                    MPI.Buffer_send(42), status.source, 3, comm.communicator
                )
                push!(sreqs, sreq)
                @debugCommPrintln("Got Eulerian request from $(status.source)")
                lock(control.locks.eulerianRequest) do
                    push!(inqRanks, status.source)
                end
            end
            if barrierOn
                comm_test(breq, barrierFlag)
            elseif MPI.Testall(sreqs)
                breq = MPI.Ibarrier(comm.communicator)
                barrierOn = true
            end
        end

        # @SiSc: changed to eulerianSreqsU; added T and rhoV
        eulerianSreqsU, eulerianSreqsT, eulerianSreqsrhoV =
            serve_eulerian(inqRanks, eulerian, comm, control)

        # Reset sources on the device to zero
        for de in reg["deviceEulerian"]
            fill!(de.UTrans, ScalarVec(0SCL, 0SCL, 0SCL)) #!!
            # @SiSc: added hTrans and rhoVTRans
            fill!(de.hTrans, 0SCL)
            fill!(de.rhoVTrans, 0SCL)
        end

        # @SiSc: changed to eulerianSreqsU; added T and rhoV
        # question @Sergey: additional waits necessary?
        for req in eulerianSreqsU comm_wait(req) end
        for req in eulerianSreqsT comm_wait(req) end
        for req in eulerianSreqsrhoV comm_wait(req) end

        for (i, _, iRank₁) in zipIter
            comm_wait(rreqsU[i])
            comm_wait(rreqsT[i])
            comm_wait(rreqsrhoV[i])
            unlock(control.locks.eulerianComms[iRank₁])
        end

        # Copy all required Eulerian to device memory
        allReqRanks₁ =
            Iterators.map(x -> x+1, comm.member.requiredEulerianRanks)
        for iRank₁ in allReqRanks₁
            lock(control.locks.eulerianComms[iRank₁]) do
                copyto!(reg["deviceEulerian"][iRank₁].U, eulerian[iRank₁].U)
                # @SiSc: added T, rhoV
                copyto!(reg["deviceEulerian"][iRank₁].T, eulerian[iRank₁].T)
                copyto!(
                    reg["deviceEulerian"][iRank₁].rhoV, eulerian[iRank₁].rhoV
                )
            end
        end

        empty!(comm.member.requiredEulerianRanks)

        # Enable garbage collector at the same time as other thread
        GC.enable(true)
        notify(control.events.U_copied)

        print("Evolve particles\n")

        for l in control.locks.eulerianComms
            lock(l)
        end
        lock(control.locks.chunkTransfers) do
            @sync begin
                for (i, chunk) in enumerate(chunks)
                    @async begin
                        # Reset the bounding box in device memory
                        hostChunkBb = BoundingBox{Vector{scalar}}(
                            fill(Inf, 3), fill(-Inf, 3)
                        )
                        copy!(chunk.boundingBox, hostChunkBb)
                        kernel(
                            chunk, reg["deviceEulerianPointer"], mesh, executor;
                            threads, blocks
                        )
                        copy!(hostChunkBb, chunk.boundingBox)
                        determine!(
                            comm.member.requiredEulerianRanks,
                            hostChunkBb,
                            mesh,
                            control
                        )

                        nothing
                    end
                end
            end
        end
        for l in control.locks.eulerianComms
            unlock(l)
        end

        tEvolve = time() - tStart
        global totalTime += tEvolve
        print("Lagrangian solver: waiting time for evolve to finish = \
            $(round(tEvolve, sigdigits=4)) s; total waiting time = \
            $(round(totalTime, sigdigits=4)) s\n"
        )

        wait(control.events.Eulerian_computed)
        GC.enable(false)

        # Correct and estimate the source
        lock(control.locks.eulerianComms[comm.jlRank]) do
            copyto!(
                eulerian[comm.jlRank].UTrans,
                reg["deviceEulerian"][comm.jlRank].UTrans
            )
            # @SiSc: added hTrans
            copyto!(
                eulerian[comm.jlRank].hTrans,
                reg["deviceEulerian"][comm.jlRank].hTrans
            )
            # @SiSc: added rhoVTrans
            copyto!(
                eulerian[comm.jlRank].rhoVTrans,
                reg["deviceEulerian"][comm.jlRank].rhoVTrans
            )
            # @SiSc: added hTrans and rhoVTrans
            estimate_source!(
                eulerian[comm.jlRank].UTrans,
                eulerian[comm.jlRank].hTrans,
                eulerian[comm.jlRank].rhoVTrans,
                control.extrapolator
            )
        end

        # @SiSc: changed to sourceRreqsU; added T and rhoV
        sourceRreqsU, sourceRreqsT, sourceRreqsrhoV,
            sourceBuffersU, sourceBuffersT, sourceBuffersrhoV =
            receive_sources(inqRanks, comm, eulerian)

        # Copy sources to CPU and send them to slaves
        # @SiSc: changed to sourceSreqsU; added T and rhoV
        sourceSreqsU = Vector{MPI.Request}(undef, nRequests)
        sourceSreqsT = Vector{MPI.Request}(undef, nRequests)
        sourceSreqsrhoV = Vector{MPI.Request}(undef, nRequests)

        for (i, iRank₀, iRank₁) in zipIter
            @debugCommPrintln("Send source to $iRank₀")
            lock(control.locks.eulerianComms[iRank₁])
            copyto!(
                eulerian[iRank₁].UTrans, reg["deviceEulerian"][iRank₁].UTrans
            )
            sourceSreqsU[i] = MPI.Isend(
                eulerian[iRank₁].UTrans, comm.communicator, dest=iRank₀, tag=1
            )
            # @SiSc: added hTrans
            copyto!(
                eulerian[iRank₁].hTrans, reg["deviceEulerian"][iRank₁].hTrans
            )
            sourceSreqsT[i] = MPI.Isend(
                eulerian[iRank₁].hTrans, comm.communicator, dest=iRank₀, tag=1
            )
            # @SiSc: added rhoVTrans
            copyto!(
                eulerian[iRank₁].rhoVTrans,
                reg["deviceEulerian"][iRank₁].rhoVTrans
            )
            sourceSreqsrhoV[i] = MPI.Isend(
                eulerian[iRank₁].rhoVTrans,
                comm.communicator, dest=iRank₀, tag=1
            )
        end

        for (i, _, iRank₁) in zipIter
            # @SiSc: changed to sourceSreqsU; added T and rhoV
            # question @Sergey: correct and necessary?
            comm_wait(sourceSreqsU[i])
            comm_wait(sourceSreqsT[i])
            comm_wait(sourceSreqsrhoV[i])
            unlock(control.locks.eulerianComms[iRank₁])
        end

        lock(control.locks.eulerianComms[comm.jlRank]) do
            # @SiSc: changed to sourceRreqsU, sourceBuffersU
            for (i, req) in enumerate(sourceRreqsU)
                comm_wait(req)
                eulerian[comm.jlRank].UTrans .+= sourceBuffersU[i]
            end
            # @SiSc: added T
            for (i, req) in enumerate(sourceRreqsT)
                comm_wait(req)
                eulerian[comm.jlRank].hTrans .+= sourceBuffersT[i]
            end
            # @SiSc: added rhoV
            # question @Sergey: correct?
            for (i, req) in enumerate(sourceRreqsrhoV)
                comm_wait(req)
                eulerian[comm.jlRank].rhoVTrans .+= sourceBuffersrhoV[i]
            end
            estimate_source!(
                eulerian[comm.jlRank].UTrans,
                eulerian[comm.jlRank].hTrans,
                eulerian[comm.jlRank].rhoVTrans,
                control.extrapolator
            )
        end

        empty!(inqRanks)

        notify(control.events.S_copied)
    end
    return nothing
end

# Drives evolve on GPU
function evolve!(control, ::GPU)
    unlock(control.locks.eulerianComms[comm.jlRank])
    if !firstPass
        notify(control.events.Eulerian_computed)
        wait(control.events.S_copied)
    end
    global firstPass = false
    wait(control.events.U_copied)
    lock(control.locks.eulerianComms[comm.jlRank])
    return nothing
end

function compute_bounding_box(chunk, executor::GPU)
    @inbounds begin
        c = chunk
        i = (blockIdx().x - 1LBL) * blockDim().x + threadIdx().x
        if(i <= c.N)
            pos = ScalarVec(c.X[i], c.Y[i], c.Z[i])
            update!(c.boundingBox, pos, executor)
        end
    end
    return nothing
end

# Evolve particles using GPU
function evolve_on_device!(chunk, eulerian, mesh, executor)
    @inbounds begin
        nSteps = 100LBL
        Δtd = chunk.time[1].Δt / nSteps  # Time step for the dispersed phase
        nParticles = chunk.N
        i = (blockIdx().x - 1LBL) * blockDim().x + threadIdx().x
        if(i <= nParticles)
            evolve_particle!(chunk, eulerian, i, Δtd, mesh, nSteps, executor)
        end
    end
    return nothing
end

@inline function evolve_particle!(
    chunk, eulerianArr, i, Δt, mesh, nSteps, executor
)
    @inbounds begin
        c = chunk
        μᶜ = c.μᶜ   # dynamic viscosity of air
        ρᵈ = c.ρᵈ   # density of water
        g = ScalarVec(c.gX, c.gY, c.gZ) # gravitational acceleration
        Cₚᶜ = c.Cₚᶜ # heat capacity of carrier phase
        Cₚᵈ = c.Cₚᵈ # heat capacity of disperse phase
        ρᶜ = c.ρᶜ   # density of air
        Dᵈᶜ = c.Dᵈᶜ # diffusivity coefficient water vapour in air
        Mᵈ = c.Mᵈ   # molar weight of water
        σᶜ = c.σᶜ   # surface tension of water in air
        RG = c.RG   # gas constant
        SLH = c.SLH # latent heat of water evapouration

        ⌀    = scalar(c.d[i])
        ⌀New = scalar(c.d[i])

        pos    = MutScalarVec(c.X[i], c.Y[i], c.Z[i])
        posNew = MutScalarVec(c.X[i], c.Y[i], c.Z[i])

        uᵈ      = MutScalarVec(c.U[i], c.V[i], c.W[i])
        uᵈNew   = MutScalarVec(c.U[i], c.V[i], c.W[i])
        uᵈNewCp = MutScalarVec(c.U[i], c.V[i], c.W[i])
        urel = MutScalarVec(0, 0, 0)

        Tᵈ      = scalar(c.T[i])
        TᵈNew   = scalar(c.T[i])

        mᵈ    = scalar(π*⌀^3*ρᵈ/6SCL)
        mᵈNew = mᵈ

        I, J, K, posI::label, partitionI::label = locate(pos, mesh)
        eulerian = eulerianArr[partitionI]
        uᶜ = eulerian.U[posI]
        Tᶜ = eulerian.T[posI]
        rhoVᶜ = max(0SCL, eulerian.rhoV[posI])

        # sources
        drhoVTrans = 0SCL
        dhTrans    = 0SCL
        dUTrans    = MutScalarVec(0, 0, 0)

        for t = 1LBL:nSteps

            ### mass ###
            # temperature in °C
            TᶜinC = scalar(Tᶜ-273.15SCL)

            # Water vapour saturation pressure (Arden-Buck-Equation) in Pa
            pvsat = scalar(611.21SCL*exp(
                (18.678SCL - TᶜinC/234.5SCL)*(TᶜinC/(257.14SCL + TᶜinC)))
            )

            # water vapour density at saturation in kg/m³
            rhovsat = scalar(Mᵈ*pvsat/(RG*Tᶜ))

            # air molecular density (43.04*Tref*p/(T*pref)) in mol/m³
            rhoMolAir = 43.04SCL*283.15SCL/(Tᶜ*1.01325SCL)

            # Water vapour pressure in Pa
            pv = rhoVᶜ/(rhoMolAir*Mᵈ)*1e5SCL

            # Saturation ratio of water vapour in continuous phase
            Sinf = pv/pvsat

            # Saturation ratio of water vapour at particle surface
            # (Only Kelvin/curvature effect, no Raoult/solute effect)
            Ssfc = scalar(exp(4SCL*Mᵈ*σᶜ/(RG*Tᵈ*ρᵈ*⌀)))

            # Integration over time using semi-implicit Euler
            # (min. particle mass equiv. to ⌀~1µm)
            mᵈNew = max(
                5e-16SCL,
                scalar(mᵈ + 2SCL*π*Dᵈᶜ*⌀*rhovsat*(Sinf - Ssfc)*Δt)
            )
            Δmᵈ = mᵈNew - mᵈ
            drhoVTrans -= Δmᵈ

            # Temperature change due to latent heat release
            Tᵈ += Δmᵈ*SLH/(0.5SCL*(mᵈ + mᵈNew)*Cₚᵈ)
            ⌀New = scalar(cbrt(6SCL*mᵈNew/(ρᵈ*π)))

            ### temperature ###
            # Thermal conductivity of air (temperature corrected)
            κᶜ = scalar(Tᵈ*8.9182E-5SCL)

            # surface values
            Tˢ = (2SCL*Tᵈ + Tᶜ)/3SCL
            TRatio = Tᶜ/Tˢ
            ρˢ = ρᶜ*TRatio
            μˢ = μᶜ/TRatio
            κˢ = κᶜ/TRatio
            Pr = Cₚᶜ*μˢ/κˢ

            # Slip velocity
            urel .= uᶜ .- uᵈ

            # Particle Reynolds number
            Re = scalar(sqrt(sum(urel.^2))*⌀*ρˢ/μˢ)

            # Particle Nusselt number
            Nu = scalar(2SCL + 0.6SCL*sqrt(Re)*cbrt(Pr))

            # surface area of a sphere
            Asᵈ = scalar(π*⌀^2SCL)

            # Heat transfer coefficient
            htc = Nu*κˢ/⌀

            # integration coefficients
            bcp  = scalar(htc*Asᵈ/(mᵈ*Cₚᵈ))
            acp  = scalar(bcp*Tᶜ)

            # effective time step
            ΔtEff = Δt/(1SCL + bcp*Δt)

            ΔTᵈ = scalar((acp - bcp*Tᵈ)*ΔtEff)
            TᵈNew   = Tᵈ + ΔTᵈ
            dhTrans -= Cₚᵈ*(mᵈNew*TᵈNew - mᵈ*Tᵈ)

            ### velocity ###

            # drag force with Schiller-Naumann drag coefficient model
            CdRe = scalar(24SCL*(1SCL + (Re^0.687SCL)*0.15SCL))
            Fd = scalar(mᵈ*0.75SCL*μᶜ*CdRe/(ρᵈ*⌀^2SCL))

            # gravity force
            Fg = ScalarVec(mᵈ.*g.*(1SCL-ρᶜ/ρᵈ))

            # integration coefficients
            acp  = ScalarVec(Fd.*uᶜ./mᵈ)
            ancp = ScalarVec(Fg./mᵈ)
            bcp  = scalar(Fd./mᵈ)

            # effective time step
            ΔtEff = Δt/(1SCL + bcp*Δt)

            # Implicit Euler time integration of particle velocity
            Δuᵈ = ScalarVec((acp .+ ancp .- bcp.*uᵈ).*ΔtEff)
            ΔuᵈNcp = ancp*Δt
            ΔuᵈCp  = Δuᵈ - ΔuᵈNcp
            uᵈNew   = uᵈ .+ Δuᵈ
            uᵈNewCp = uᵈ .+ ΔuᵈCp
            dUTrans.x -= mᵈNew*uᵈNewCp.x - mᵈ*uᵈ.x
            dUTrans.y -= mᵈNew*uᵈNewCp.y - mᵈ*uᵈ.y
            dUTrans.z -= mᵈNew*uᵈNewCp.z - mᵈ*uᵈ.z

            # update the position
            posNew .= pos .+ uᵈNew.*Δt

            # Find the new position index and check whether the parcel hits a
            # boundary.  If yes, then hold the corresponding position component
            # and mirror the velocity component.  In this way, the parcel
            # will always stay within the domain.

            I, J, K, posNewI::label, partitionNewI::label = locate(posNew, mesh)

            # Has the parcel moved to another cell or partition?
            if (posNewI != posI) || (partitionNewI != partitionI)

                # Has the parcel hit a boundary?
                bx = (I < 0LBL) || (I >= mesh.N.x)
                by = (J < 0LBL) || (J >= mesh.N.y)
                bz = (K < 0LBL) || (K >= mesh.N.z)

                if bx
                    posNew.x = pos.x
                    uᵈNew.x = -uᵈ.x
                end

                if by
                    posNew.y = pos.y
                    uᵈNew.y = -uᵈ.y
                end

                if bz
                    posNew.z = pos.z
                    uᵈNew.z = -uᵈ.z
                end

                # Reevaluate the cell and partition index the parcel is in,
                # since it may have changed after the boundary hit.
                if (bx || by || bz)
                    I, J, K, posNewI, partitionNewI = locate(posNew, mesh)
                end

                if (partitionNewI != partitionI)
                    # Reset pointer to eulerian
                    eulerian = eulerianArr[partitionI]
                end

                if (t != nSteps && posNewI != posI)
                    update!(
                        eulerian, dUTrans, dhTrans, drhoVTrans, posI, executor
                    )
                    dUTrans.x = 0
                    dUTrans.y = 0
                    dUTrans.z = 0
                    drhoVTrans = 0SCL
                    dhTrans = 0SCL
                    uᶜ = eulerian.U[posNewI]
                    Tᶜ = eulerian.T[posNewI]
                    rhoVᶜ = eulerian.rhoV[posNewI]
                    posI = posNewI
                end
            end

            pos .= posNew
            uᵈ .= uᵈNew
            Tᵈ = TᵈNew
            ⌀ = ⌀New
            mᵈ = mᵈNew
        end

        update!(eulerian, dUTrans, dhTrans, drhoVTrans, posI, executor)

        c.X[i] = pos.x
        c.Y[i] = pos.y
        c.Z[i] = pos.z
        c.U[i] = uᵈ.x
        c.V[i] = uᵈ.y
        c.W[i] = uᵈ.z
        c.T[i] = Tᵈ
        c.d[i] = ⌀

        update!(c.boundingBox, posNew, executor)
    end
    return nothing
end

function write(chunk, ::CPU)
    c = chunk
    t = round(c.time[1].t, sigdigits=4)
    println("Writing time ", c.time[1].t, " as ", t)

    if !(haskey(reg, "paraview"))
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i, )) for i = 1:c.N]
        pvd = paraview_collection("particleTimeSeries")
        reg["paraview"] = Dict("cells" => cells, "pvd" => pvd)
    end

    pv = reg["paraview"]
    cells = pv["cells"]
    pvd = pv["pvd"]
    if !isdir("dataVTK")
        mkdir("dataVTK")
    end
    vtk_grid("dataVTK/particleFields_$(t).vtu", c.X, c.Y, c.Z, cells) do vtk
        vtk["U", VTKPointData()] = transpose(stack([c.U, c.V, c.W]))
        vtk["d", VTKPointData()] = c.d
        vtk["T", VTKPointData()] = c.T # @SiSc: added temperature field
        pvd[t] = vtk
    end
    return nothing
end

function write(chunk, ::GPU)
    c = chunk
    if !(haskey(reg, "hostChunk"))
        reg["hostChunk"] = allocate_chunk
        (
            CPU(), c.N, c.μᶜ, c.ρᶜ, c.ρᵈ, c.gX, c.gY, c.gZ, c.Cₚᶜ, c.Cₚᵈ, c.Dᵈᶜ,
            c.Mᵈ, c.σᶜ, c.RG, c.SLH
        ) # @SiSc: added additional properties
    end
    lock(control.locks.chunkTransfers) do
        copy!(reg["hostChunk"], c)
    end
    write(reg["hostChunk"], CPU())
    return nothing
end

function write(chunks, comm::Comm{Master}, executor)
    GC.enable(false)
    # TODO Enable writing for all chunks, not only the first one
    if comm.isMaster write(first(chunks), executor) end
    GC.enable(true)
end

function write(chunk, ::Comm{<:CommMember}, executor) end

function write_paraview_collection(::Comm{Master})
    GC.enable(false)
    if haskey(reg, "paraview")
        if haskey(reg["paraview"], "pvd")
            vtkCollection = reg["paraview"]["pvd"]
            println("Writing paraview collection to ", vtkCollection.path)
            flush(stdout)
            vtk_save(vtkCollection)
            return nothing
        end
    end
    println("No paraview collection written since no paraview instance found"
            *" in the registry")
    GC.enable(true)
    return nothing
end

function write_paraview_collection(::Comm{<:CommMember}) end


###############################################################################
# Coupling to C++

function allocate_array_j(
        name::Cstring, size::Cint, nComponents::Cint, typeByteSize::Cint
    )::Ptr{Cdouble}
    GC.enable(false)

    GC.@preserve name nameSymbol = Symbol(unsafe_string(pointer(name)))
    print("Allocating $nameSymbol\n")

    if Int(typeByteSize) != Int(sizeof(scalar))
        throw(
            ErrorException(
                string(
                    "Size of Julia type (in Bytes): ", sizeof(scalar),
                    " is different from the extern type: ", typeByteSize
                )
            )
        )
    end

    if Int(size) != Int(prod(mesh.partitionN))
        throw(
            ErrorException(
                string(
                    "Size of Julia mesh: ", prod(mesh.partitionN),
                    " is different from field's ", nameSymbol, " size: ",
                    Int(size)
                )
            )
        )
    end

    # By now, assume that the requested field already has been allocated
    field = getfield(reg["eulerian"][comm.jlRank], nameSymbol)
    # nComponents needs to be Int
    # n = Int(nComponents)
    # if (n == 1)
    #     field = Vector{scalar}(undef, size)
    # else
    #     field = Vector{SVector{n, scalar}}(undef, size)
    # end
    GC.@preserve field cFieldPtr = Base.unsafe_convert(Ptr{Cdouble}, field)
    print("Allocating $nameSymbol done\n")
    GC.enable(true)
    return cFieldPtr
end

const allocate_array_ptr =
    @cfunction(allocate_array_j, Ptr{Cdouble}, (Cstring, Cint, Cint, Cint))

function evolve_cloud(Δt)
    # Workaround for the segmentation fault error when using multiple Julia
    # threads.  The error occurs when the garbage collector (GC) is triggered
    # in more than one thread at the same time.  Thus, trigger GC manually and
    # then disable it in the affected functions.
    global reg
    GC.enable(true)
    reg["timestepsSinceLastGC"] += 1
    if reg["timestepsSinceLastGC"] >= reg["noGcTimestepInterval"]
        gcDiff = Base.GC_Diff(Base.gc_num(), reg["gc_num"])
        reg["gc_num"] = Base.gc_num()
        tNow = time()
        GC.gc()
        reg["timestepsSinceLastGC"] = 0
        timing(tNow, "Run garbage collection")
        println("Allocated since last GC: $(gcDiff.allocd/1e6) MB")
    end
    GC.enable(false)

    if comm.isMaster
        print("Evolve cloud\n")
        for chunk in chunks
            increment_time!(chunk, Δt, comm, executor)
        end
    end

    global tStart = time()
    evolve!(control, executor)

    minrhoVTrans = minimum(reg["eulerian"][comm.jlRank].rhoVTrans)
    maxrhoVTrans = maximum(reg["eulerian"][comm.jlRank].rhoVTrans)
    @debugCommPrintln("min/max rhoVTrans: $minrhoVTrans; $maxrhoVTrans")

    # Enable GC
    GC.enable(true)
    return nothing
end

const evolve_cloud_ptr = @cfunction(evolve_cloud, Cvoid, (Cdouble,))

###############################################################################
GC.enable(false)

ΔtInitMethods = time() - ΔtLoadModules

# Global data init
executor = GPU()
# executor = CPU()

const comm = initComm(executor)
println("Loaded modules in ", round(ΔtLoadModules; sigdigits=4), " s")
println("Initialized methods in ", round(ΔtInitMethods; sigdigits=4), " s")
tNow = timing(ΔtInitMethods, "Initialized communication")

println("Julia active project: ", Base.active_project())

nParticles = 50_000_000
nChunks = 1
nPTotal = nChunks*nParticles

# constants in SI units
# @SiSc: added additional properties
μᶜ = 1.8e-5SCL              # continuous phase dynamic viscosity
ρᶜ = 1.2SCL                 # continuous phase density
ρᵈ = 1000.0SCL              # disperse phase density
g = ScalarVec(0, 0, 0)      # gravitational acceleration
Cₚᶜ = 1000SCL               # specific heat capacity of air
Cₚᵈ = 4200SCL               # specific heat capacity of water
Dᵈᶜ = 24.5e-6SCL            # diffusivity coefficient of water vapour in air
Mᵈ = 18.01528e-3SCL         # molar mass of water
σᶜ = 72.8e-3SCL             # surface tension of water in air
RG = 8.3144598SCL           # gas constant
SLH = 2264.71SCL            # specific latent heat of water vaporisation

nCellsPerDirection = [32, 32, 32]
nCells = prod(nCellsPerDirection)
origin = [0.0, 0.0, 0.0]
ending = [1.0, 1.0, 1.0]

if comm.size == 1
    # Serial run
    decomposition = (1, 1, 1)
else
    # Decomposition coefficients need to be the same as in the decomposeParDict
    decomposition = (4, 4, 2)
end

reg["noGcTimestepInterval"] = 100
reg["timestepsSinceLastGC"] = 100

# @SiSc: added some additional output
@show nParticles nChunks nPTotal nCellsPerDirection nCells origin ending decomposition
@show μᶜ ρᶜ ρᵈ g Cₚᶜ Cₚᵈ Dᵈᶜ Mᵈ σᶜ RG SLH
@show reg["noGcTimestepInterval"]

mesh = construct_mesh(nCellsPerDirection, origin, ending, decomposition)
tNow = timing(tNow, "Initialized mesh")
@show mesh.N mesh.origin mesh.ending mesh.L mesh.Δ mesh.rΔ
@show mesh.decomposition mesh.partitionN mesh.partitionNxTimesNy
control = Control(
    Locks(comm.size), Events(), ConstExtrapolator(prod(mesh.partitionN))
)
tNow = timing(tNow, "Initialized control")

# Allocate Eulerian fields
# @SiSc: adapted for additional fields
reg["eulerian"] = Vector{TwoWayEulerian{VectorField, ScalarField}}(
    undef, comm.size
)
tNow = timing(tNow, "Allocated Eulerian ranks array")

tStart = time()
totalTime = 0.0
firstPass = true

# Lock own Eulerian to put async evolve on wait until it's notified by call to
# evolve_cloud from OF
lock(control.locks.eulerianComms[comm.jlRank])

if comm.isHost
    chunks = Vector{Chunk}(undef, nChunks)
    for i in eachindex(chunks)
        chunks[i] = allocate_chunk(
            comm, executor, nParticles,
            μᶜ, ρᶜ, ρᵈ, g.x, g.y, g.z, Cₚᶜ, Cₚᵈ, Dᵈᶜ, Mᵈ, σᶜ, RG, SLH
        ) # @SiSc: added additional properties
        init!(chunks[i], mesh, executor, i)
    end
    tNow = timing(tNow, "Initialized particle chunk")

    # Write initial state
    # write(chunk, comm, executor)
    tNow = timing(tNow, "Written VTK data")

    # Allocate Eulerian for all ranks on master
    for i in eachindex(reg["eulerian"])
        reg["eulerian"][i] = TwoWayEulerian{VectorField, ScalarField}(
            prod(mesh.partitionN)
        ) # @SiSc: added ScalarField
    end

    errormonitor(
        @spawn init_async_evolve!(
            chunks, reg["eulerian"], mesh, control, comm, executor
        )
    )
    tNow = timing(tNow, "Initialized asynchronous evolve")
else
    # Allocate Eulerian only for self on slaves
    reg["eulerian"][comm.jlRank] = TwoWayEulerian{VectorField, ScalarField}(
        prod(mesh.partitionN)
    ) # SiSc: added ScalarField

    errormonitor(
        @spawn init_async_evolve!(reg["eulerian"], control, comm, executor)
    )
end

# Test few time steps when running in REPL
if isinteractive()
    # Create random velocity field
    for eulerian in reg["eulerian"]
        init_random!(eulerian.U, 2SCL, -1SCL)
        init_random!(eulerian.T, 0SCL, 293.15SCL)
        init_random!(eulerian.rhoV, 0SCL, 17.27e-3SCL)
    end # @SiSc: added T and rhoV
    tNow = timing(tNow, "Initialized velocity field")

    evolve_cloud(1e-3)
    evolve_cloud(1e-3)
end

gcDiff = Base.GC_Diff(Base.gc_num(), reg["gc_num"])
reg["gc_num"] = Base.gc_num()
println("Julia allocations during startup: $(gcDiff.allocd/1e6) MB")

println("Load asyncParallelTracking done\n")
flush(stdout)
flush(stderr)

###############################################################################
