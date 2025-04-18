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
using HilbertSpaceFillingCurve
using Statistics

ΔtLoadModules = time() - tNow

###############################################################################
# Compile-time globals

# Debug communication logging.  If true, Log to stdout from every rank prefixed
# by [rank].
DebugComm = false

###############################################################################
# Auxiliary Methods

function timing(t, s)
    timePoint = time()
    dt = timePoint - t
    print("$s in $(fmt_time(dt)) s\n")

    # Invoke flush to ensure immediate printing even when executed within C code
    flush(stdout)
    return timePoint
end

function fmt_time(t)
    return round(t, sigdigits=4)
end

function save_timings(comm, reg)
    paddedNCores = lpad(string(comm.size), 4, '0')
    open("stats_np$paddedNCores","w") do io
        println(io, "\
            nTimeSteps \
            tAllTotal tEulerTotal tEvolveTotal tDeviceComputeTotal \
            tAllMean tEulerMean tEvolveMean tDeviceComputeMean \
            tAllStd tEulerStd tEvolveStd tDeviceComputeStd\
        ")
        # Timings may have different count deu to async execution
        nTimes = min(length(reg["dtVecEuler"]), length(reg["dtVecEvolve"]))
        dtAll = first(reg["dtVecAll"], nTimes)
        dtEul = first(reg["dtVecEuler"], nTimes)
        dtEvo = first(reg["dtVecEvolve"], nTimes)
        dtDev = first(reg["dtVecDeviceCompute"], nTimes)
        timings = scalar[
            sum(dtAll),
            sum(dtEul),
            sum(dtEvo),
            sum(dtDev),
            mean(dtAll),
            mean(dtEul),
            mean(dtEvo),
            mean(dtDev),
            std(dtAll),
            std(dtEul),
            std(dtEvo),
            std(dtDev)
        ]
        print(io, nTimes)
        for t in timings
            print(io, " $(fmt_time(t))")
        end
        print(io, "\n")
    end
    return nothing
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

const VectorField = Vector{ScalarVec}
Adapt.@adapt_structure VectorField

struct TwoWayEulerian{T}
    N::label
    U::T
    UTrans::T
end

function TwoWayEulerian{T}(N) where {T}
    TwoWayEulerian{T}(
        N,
        T(undef, N),
        T(undef, N)
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
    GC_finished::Event
    GC_enabled::Event

    Events() =
        new(Event(true), Event(true), Event(true), Event(true), Event(true))
end

# Extrapolation of sources
abstract type AbstractExtrapolator end
struct ConstExtrapolator <: AbstractExtrapolator
    prevUTrans::VectorField
end

function ConstExtrapolator(N::Real)
    vf = VectorField(undef, N)
    fill!(vf, ScalarVec(0, 0, 0))
    ConstExtrapolator(vf)
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

# Superscripts: c - carrier phase; d - dispersed phase
struct Chunk{T, A}
    N::label
    μᶜ::scalar
    ρ::scalar
    ρᵈByρᶜ::scalar
    boundingBox::BoundingBox{T}
    time::A
    d::T
    X::T
    Y::T
    Z::T
    U::T
    V::T
    W::T
end

function Chunk{T, A}(N, μᶜ, ρ, ρᵈByρᶜ) where {T, A}
    Chunk{T, A}(
        N,
        μᶜ,
        ρ,
        ρᵈByρᶜ,
        BoundingBox{T}(),
        A(undef, 1),
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
    hostCommunicator::MPI.Comm
    hostCommSize::label
end

Master(deviceNumber, hostCommunicator) =
    Master(
        deviceNumber, Set{label}(), Vector{label}(), hostCommunicator,
        MPI.Comm_size(hostCommunicator)
    )

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
    nRanksPerNode = MPI.Comm_size(shmComm)
    hostStride = nRanksPerNode ÷ nDevicesPerNode
    # Prevent 0 stride
    hostStride = hostStride < 1 ? 1 : hostStride

    # Assign devices to ranks uniformly
    hostRanks =
        range(0, step=hostStride, length=min(nRanksPerNode, nDevicesPerNode))
    if shmRank in hostRanks
        hostComm = MPI.Comm_split(MPI.COMM_WORLD, 1, 0)
        deviceNumber = shmRank ÷ hostStride
        CUDA.device!(deviceNumber)
        comm = Comm(Master(deviceNumber, hostComm), MPI.COMM_WORLD)
    else
        hostComm = MPI.Comm_split(MPI.COMM_WORLD, nothing, 0)
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

function allocate!(chunks, ::Comm{Master}, executor, constructorArgs...)
    for i in eachindex(chunks)
        chunks[i] = allocate_chunk(executor, constructorArgs...)
    end
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

function init!(chunks, mesh, executor, randSeed=19891)
    for i in eachindex(chunks)
        chunks[i] = allocate_chunk(comm, executor, nParticlesPerChunk, μᶜ, ρᵈ, ρᵈ/ρᶜ)
        c = chunks[i]
        set_time!(c, 0.0, 0.0, executor)

        rng = default_rng(executor)
        Random.seed!(rng, randSeed)
        fill!(c.boundingBox.min, 0.0)
        fill!(c.boundingBox.max, 0.0)
        rand!(rng, c.d)
        rand!(rng, c.X)
        rand!(rng, c.Y)
        rand!(rng, c.Z)
        @. c.d = c.d*5e-3SCL + 5e-3SCL
        @. c.X = c.X*mesh.L.x + mesh.origin.x
        @. c.Y = c.Y*mesh.L.y + mesh.origin.y
        @. c.Z = c.Z*mesh.L.z + mesh.origin.z
        # Clustered init in a part of the domain
        # @. c.X = c.X*mesh.Δ.x*mesh.partitionN.x*0.5*randSeed + mesh.origin.x + 0.6*mesh.L.x
        # @. c.Y = c.Y*mesh.Δ.y*mesh.partitionN.y*0.5*randSeed + mesh.origin.y + 0.6*mesh.L.y
        # @. c.Z = c.Z*mesh.Δ.z*mesh.partitionN.z*0.5*randSeed + mesh.origin.z + 0.6*mesh.L.z
        fill!(c.U, 0.0)
        fill!(c.V, 0.0)
        fill!(c.W, 0.0)
    end

    return nothing
end

const bitmask_t = HilbertSpaceFillingCurve.bitmask_t

function hilbert!(px, py, pz, j, p, d::T, ndims, nbits = 32) where T <: Integer
    @assert ndims*nbits <= sizeof(bitmask_t) * HilbertSpaceFillingCurve.bits_per_byte

    ccall(
        (:hilbert_i2c, HilbertSpaceFillingCurve.libhilbert), Nothing,
        (Int, Int, bitmask_t, Ptr{bitmask_t}), ndims, nbits, d, p
    )
    px[j] = p[1]
    py[j] = p[2]
    pz[j] = p[3]
    return nothing
end

function initWithHilbert!(
    chunks, mesh, comm, nParticlesPerChunk, ::GPU, randSeed=19891
)
    nChunksGlobal = length(chunks)
    startChunk = 0
    hostCommSize = MPI.Comm_size(comm.member.hostCommunicator)
    @show hostCommSize
    if(hostCommSize > 1)
        startChunk = MPI.Exscan(nChunksGlobal, +, comm.member.hostCommunicator)
        nChunksGlobal = MPI.Bcast(
            startChunk + nChunksGlobal,
            hostCommSize - 1,
            comm.member.hostCommunicator
        )
        @show startChunk nChunksGlobal
    end

    nHilbert = 10
    lHilbert = (2^nHilbert)^3

    p_CPU_x = Array{Int}(undef, nParticlesPerChunk)
    p_CPU_y = Array{Int}(undef, nParticlesPerChunk)
    p_CPU_z = Array{Int}(undef, nParticlesPerChunk)
    p_GPU_x = CuArray{Int}(undef, nParticlesPerChunk)
    p_GPU_y = CuArray{Int}(undef, nParticlesPerChunk)
    p_GPU_z = CuArray{Int}(undef, nParticlesPerChunk)

    for i in eachindex(chunks)
        c = chunks[i]

        set_time!(c, 0.0, 0.0, executor)

        rng = default_rng(executor)
        Random.seed!(rng, randSeed)
        fill!(c.boundingBox.min, 0.0)
        fill!(c.boundingBox.max, 0.0)
        # rand! produces a random number within range [0, 1)
        rand!(rng, c.d)
        rand!(rng, c.X)
        rand!(rng, c.Y)
        rand!(rng, c.Z)
        @. c.d = c.d*5e-3SCL + 5e-3SCL


        hStart = round(Int, (startChunk + (i - 1)*lHilbert)/nChunksGlobal)
        hEnd = round(Int, (startChunk + i*lHilbert)/nChunksGlobal)
        # The parent function is not type stable.  Put the busy loop into a
        # function to prevent allocations.
        function run_hilbert()
            p = HilbertSpaceFillingCurve.bitmask_t.(zeros(3))
            for j in 1:c.N
                hilbert!(
                    p_CPU_x, p_CPU_y, p_CPU_z, j, p, rand(hStart:hEnd), 3,
                    nHilbert
                )
            end
        end
        run_hilbert()

        copyto!(p_GPU_x, p_CPU_x)
        copyto!(p_GPU_y, p_CPU_y)
        copyto!(p_GPU_z, p_CPU_z)

        @. c.X = (
            mesh.origin.x + mesh.Δ.x*((mesh.N.x*p_GPU_x)÷2^nHilbert + c.X)
        )*0.99999SCL  # prevent position at a domain boundary
        @. c.Y = (
            mesh.origin.y + mesh.Δ.y*((mesh.N.y*p_GPU_y)÷2^nHilbert + c.Y)
        )*0.99999SCL
        @. c.Z = (
            mesh.origin.z + mesh.Δ.z*((mesh.N.z*p_GPU_z)÷2^nHilbert + c.Z)
        )*0.99999SCL

        fill!(c.U, 0.0)
        fill!(c.V, 0.0)
        fill!(c.W, 0.0)
    end

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

@inline function set_parcel_state(::TwoWayEulerian, velocity)
    return ScalarVec(velocity)
end

@inline function update!(
    eulerian::TwoWayEulerian, state0, velocity, posI, mᵈByρᶜ, ::CPU
)
    @inbounds eulerian.UTrans[posI] += mᵈByρᶜ*(state0 - velocity)
    return set_parcel_state(eulerian, velocity)
end

@inline function update!(
    eulerian::TwoWayEulerian, state0, velocity, posI, mᵈByρᶜ, ::GPU
)
    @inbounds  begin
        for i=1LBL:3LBL
            # ScalarVec is immutable and thus its components cannot be mutated
            # atomically the straightforward way.  Reinterpret the eulerian
            # field at a location specified by posI with an offset as scalar
            # and get pointer to it.  Use the pointer to mutate data.
            scalar_ptr = pointer(
                reinterpret(scalar, eulerian.UTrans), (posI-1LBL)*3LBL+i
            )
            CUDA.atomic_add!(
                scalar_ptr, mᵈByρᶜ*(state0[i] - velocity[i])
            )
        end
    end
    return set_parcel_state(eulerian, velocity)
end

# Apply the change in velocity to the source due to bounce at boundary
@inline function update_at_boundary!(
    state0, ::TwoWayEulerian, velocity, componentI
)
    @inbounds @reset state0[componentI] -= 2SCL*velocity[componentI]
    return state0
end

# Constant extrapolator assumes that the current source is the same as the true
# from the previous time step.  The estimated source from the previous time
# step is corrected using the true source from the previous time step.
# estSⁿ = trueSⁿ⁻¹ - estSⁿ⁻¹ + extrapSⁿ, with extrapSⁿ = trueSⁿ⁻¹
function estimate_source!(currUTrans, extrapolator::ConstExtrapolator)
    currUTrans .= 2.0.*currUTrans .- extrapolator.prevUTrans
    for i in eachindex(extrapolator.prevUTrans)
        @inbounds extrapolator.prevUTrans[i] = currUTrans[i]
    end
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
            li[minP[1]:maxP[1], minP[2]:maxP[2], minP[3]:maxP[3]]
            .- 1  # Rank indexing is zero-based
        )
    end
end

# Evolve particles using CPU
function evolve!(chunk, eulerian, mesh, Δt, control, executor::CPU)
    @inbounds begin
        nSteps = 10LBL
        ΔtP = chunk.time[1].Δt / nSteps
        for i = 1LBL:chunk.N
            evolve_particle!(chunk, eulerian, i, ΔtP, mesh, nSteps, executor)
        end

        tEvolve = time() - ["tEvolveStart"]
        if reg["iTimeStep"] > reg["nSkipInitTimeStepsProfiling"]
            reg["tEvolveTotal"] += tEvolve
            push!(reg["dtVecEvolve"], tEvolve)
        end
        print("Lagrangian solver timings: current evolve = \
            $(fmt_time(tEvolve)), s; total time = \
            $(fmt_time(reg["tEvolveTotal"])) s\n"
        )
    end
    return nothing
end

function serve_eulerian(inquiringEulerianRanks, eulerian, comm, control)
    sreqs = Vector{MPI.Request}(undef, length(inquiringEulerianRanks))
    for (i, inqHost) in enumerate(inquiringEulerianRanks)
        lock(control.locks.eulerianComms[comm.jlRank]) do
            @debugCommPrintln("Send U to $inqHost")
            sreqs[i] = MPI.Isend(
                eulerian[comm.jlRank].U, comm.communicator, dest=inqHost,
                tag=0
            )
        end
    end
    return sreqs
end

function receive_sources(inquiringEulerianRanks, comm, eulerian)
    inqRanks = inquiringEulerianRanks
    sourceRreqs = Vector{MPI.Request}(undef, length(inqRanks))
    sourceBuffers = Vector{VectorField}(undef, length(inqRanks))
    if !isempty(inqRanks)
        for (i, inqRank) in enumerate(inqRanks)
            @debugCommPrintln("Receive source from $inqRank")
            sourceBuffers[i] = similar(eulerian[comm.jlRank].UTrans)
            sourceRreqs[i] = MPI.Irecv!(
                sourceBuffers[i], comm.communicator; source=inqRank, tag=1
            )
        end
    end
    return (sourceRreqs, sourceBuffers)
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

        for req in sreqs comm_wait(req) end
        eulerianSreqs = serve_eulerian(inqRanks, eulerian, comm, control)
        for req in eulerianSreqs comm_wait(req) end

        # Enable garbage collector at the same time as other thread
        if reg["timeStepsSinceLastGC"] >= reg["noGcTimeStepInterval"]
            GC.enable(true)
            notify(control.events.GC_enabled)
            wait(control.events.GC_finished)
            GC.enable(false)
        end
        notify(control.events.U_copied)
        wait(control.events.Eulerian_computed)

        sourceRreqs, sourceBuffers = receive_sources(inqRanks, comm, eulerian)
        lock(control.locks.eulerianComms[comm.jlRank]) do
            for (i, req) in enumerate(sourceRreqs)
                comm_wait(req)
                eulerian[comm.jlRank].UTrans .+= sourceBuffers[i]
            end
        end

        estimate_source!(eulerian[comm.jlRank].UTrans, control.extrapolator)
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
            Vector{TwoWayEulerian{CuVector{ScalarVec}}}(undef, comm.size)
        # CUDA container for the pointers (CuDeviceVector) to the Eulerian
        # containers.  It needs to be stored separately since
        # CuVector{CuVector} is not possible.
        reg["deviceEulerianPointer"] =
            CuVector{TwoWayEulerian{CuDeviceVector{ScalarVec, 1}}}(
                undef, comm.size
            )
        for i in eachindex(reg["deviceEulerian"])
            # Initialize Eulerian fields
            reg["deviceEulerian"][i] =
                TwoWayEulerian{CuVector{ScalarVec}}(eulerian[i].N)
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

    iTimeStep = 0

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
        rreqs = Vector{MPI.Request}(undef, nRequests)

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
            rreqs[i] = MPI.Irecv!(
                eulerian[iRank₁].U, comm.communicator, source=iRank₀, tag=0
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

        eulerianSreqs = serve_eulerian(inqRanks, eulerian, comm, control)

        # Reset sources on the device to zero
        for de in reg["deviceEulerian"]
            fill!(de.UTrans, ScalarVec(0SCL, 0SCL, 0SCL))  #!!
        end

        for req in eulerianSreqs comm_wait(req) end

        for (i, _, iRank₁) in zipIter
            comm_wait(rreqs[i])
            unlock(control.locks.eulerianComms[iRank₁])
        end

        # Copy all required Eulerian to device memory
        allReqRanks₁ =
            Iterators.map(x -> x+1, comm.member.requiredEulerianRanks)
        for iRank₁ in allReqRanks₁
            lock(control.locks.eulerianComms[iRank₁]) do
                copyto!(reg["deviceEulerian"][iRank₁].U, eulerian[iRank₁].U)
            end
        end

        empty!(comm.member.requiredEulerianRanks)

        # Enable garbage collector at the same time as other thread
        if reg["timeStepsSinceLastGC"] >= reg["noGcTimeStepInterval"]
            GC.enable(true)
            notify(control.events.GC_enabled)
            wait(control.events.GC_finished)
            GC.enable(false)
        end
        notify(control.events.U_copied)

        tDeviceCompute = time()

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

        dtDeviceCompute = time() - tDeviceCompute
        tEvolve = time() - reg["tEvolveStart"]
        iTimeStep += 1
        if iTimeStep > reg["nSkipInitTimeStepsProfiling"]
            reg["tEvolveTotal"] += tEvolve
            push!(reg["dtVecEvolve"], tEvolve)
            push!(reg["dtVecDeviceCompute"], dtDeviceCompute)
            reg["tDeviceComputeTotal"] += dtDeviceCompute
        end
        print("Lagrangian solver: device compute time = \
            $(fmt_time(dtDeviceCompute)) s; \
            evolve time incl. communication = \
            $(fmt_time(tEvolve)) s; total evolve time = \
            $(fmt_time(reg["tEvolveTotal"])) s\n"
        )

        wait(control.events.Eulerian_computed)

        reg["tEvolveStart"] = time()

        # Correct and estimate the source
        lock(control.locks.eulerianComms[comm.jlRank]) do
            copyto!(
                eulerian[comm.jlRank].UTrans,
                reg["deviceEulerian"][comm.jlRank].UTrans
            )
            estimate_source!(eulerian[comm.jlRank].UTrans, control.extrapolator)
        end

        sourceRreqs, sourceBuffers = receive_sources(inqRanks, comm, eulerian)

        # Copy sources to CPU and send them to slaves
        sourceSreqs = Vector{MPI.Request}(undef, nRequests)

        for (i, iRank₀, iRank₁) in zipIter
            @debugCommPrintln("Send source to $iRank₀")
            lock(control.locks.eulerianComms[iRank₁])
            copyto!(
                eulerian[iRank₁].UTrans, reg["deviceEulerian"][iRank₁].UTrans
            )
            sourceSreqs[i] = MPI.Isend(
                eulerian[iRank₁].UTrans, comm.communicator, dest=iRank₀, tag=1
            )
        end

        for (i, _, iRank₁) in zipIter
            comm_wait(sourceSreqs[i])
            unlock(control.locks.eulerianComms[iRank₁])
        end

        lock(control.locks.eulerianComms[comm.jlRank]) do
            for (i, req) in enumerate(sourceRreqs)
                comm_wait(req)
                eulerian[comm.jlRank].UTrans .+= sourceBuffers[i]
            end
            estimate_source!(eulerian[comm.jlRank].UTrans, control.extrapolator)
        end

        empty!(inqRanks)

        notify(control.events.S_copied)
    end
    return nothing
end

function collect_garbage_with_stats()
    GC.enable(true)
    gcDiff = Base.GC_Diff(Base.gc_num(), reg["gc_num"])
    reg["gc_num"] = Base.gc_num()
    println("Allocated since last GC: $(gcDiff.allocd/1e6) MB")
    tNow = time()
    # Fix: call gc with preceding sleep often results in segfault
    sleep(1e-2)
    GC.gc()
    timing(tNow, "Run garbage collection")
    reg["timeStepsSinceLastGC"] = 0
    GC.enable(false)
end

# Drives evolve on GPU
function evolve!(control, ::GPU)
    unlock(control.locks.eulerianComms[comm.jlRank])
    if reg["iTimeStep"] > 1
        notify(control.events.Eulerian_computed)
        wait(control.events.S_copied)
        if reg["timeStepsSinceLastGC"] >= reg["noGcTimeStepInterval"]
            wait(control.events.GC_enabled)
            collect_garbage_with_stats()
            notify(control.events.GC_finished)
        end
    end
    wait(control.events.U_copied)
    # Update after a synchronization point to prevent race condition
    reg["timeStepsSinceLastGC"] += 1
    lock(control.locks.eulerianComms[comm.jlRank])
    return nothing
end

# Finishes device kernels and blocks starts of new kernels.  To be run at Julia
# exit.
function finalize_evolve()
    if comm.isMaster
        println("Saving timings")
        save_timings(comm, reg)
    end
    unlock(control.locks.eulerianComms[comm.jlRank])
    # Acquiring this lock ensures that no kernels are running that would throw
    # an error when aborted
    lock(control.locks.chunkTransfers)
    notify(control.events.Eulerian_computed)
    wait(control.events.S_copied)
    if reg["timeStepsSinceLastGC"] >= reg["noGcTimeStepInterval"]
        wait(control.events.GC_enabled)
        notify(control.events.GC_finished)
    end
    wait(control.events.U_copied)
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
        nSteps = 10LBL
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
        μᶜ = c.μᶜ
        ρᵈ = c.ρ
        ρᵈByρᶜ = c.ρᵈByρᶜ
        ⌀ = c.d[i]
        mᵈByρᶜ = ρᵈByρᶜ*π*⌀^3/6SCL
        pos = MutScalarVec(c.X[i], c.Y[i], c.Z[i])
        vel = MutScalarVec(c.U[i], c.V[i], c.W[i])
        posNew = MutScalarVec(c.X[i], c.Y[i], c.Z[i])
        velNew = MutScalarVec(c.U[i], c.V[i], c.W[i])

        I, J, K, posI::label, partitionI::label = locate(pos, mesh)
        eulerian = eulerianArr[partitionI]
        uᶜ = eulerian.U[posI]
        state0 = set_parcel_state(eulerian, vel)

        for t = 1LBL:nSteps

            # Implicit Euler time integration with Stokes drag
            dragFactor = 18SCL*μᶜ*Δt/(ρᵈ*⌀^2)
            velNew .= (vel .+ dragFactor.*uᶜ)./(1 + dragFactor)
            posNew .= pos .+ velNew.*Δt

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
                    velNew.x = -vel.x
                    state0 = update_at_boundary!(state0, eulerian, vel, 1)
                end

                if by
                    posNew.y = pos.y
                    velNew.y = -vel.y
                    state0 = update_at_boundary!(state0, eulerian, vel, 2)
                end

                if bz
                    posNew.z = pos.z
                    velNew.z = -vel.z
                    state0 = update_at_boundary!(state0, eulerian, vel, 3)
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
                    state0 = update!(
                        eulerian, state0, velNew, posI, mᵈByρᶜ, executor
                    )
                    uᶜ = eulerian.U[posNewI]
                    posI = posNewI
                end
            end

            vel .= velNew
            pos .= posNew
        end

        c.X[i] = posNew.x
        c.Y[i] = posNew.y
        c.Z[i] = posNew.z
        c.U[i] = velNew.x
        c.V[i] = velNew.y
        c.W[i] = velNew.z
        update!(eulerian, state0, velNew, posI, mᵈByρᶜ, executor)
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
        pvd[t] = vtk
    end
    return nothing
end

function write(chunk, ::GPU)
    c = chunk
    if !(haskey(reg, "hostChunk"))
        reg["hostChunk"] = allocate_chunk(CPU(), c.N, c.μᶜ, c.ρ, c.ρᵈByρᶜ)
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
    global reg, comm

    # Timings
    reg["iTimeStep"] += 1
    dtVec = time() - reg["tAllStart"]
    dtEuler = time() - reg["tEulerStart"]
    reg["tAllStart"] = time()
    timing(reg["tEulerStart"], "Computed Eulerian phase")
    # Exclude predefined number of initial time steps
    if reg["iTimeStep"] > reg["nSkipInitTimeStepsProfiling"]
        reg["tAllTotal"] += dtVec
        reg["tEulerTotal"] += dtEuler
        push!(reg["dtVecAll"], dtVec)
        push!(reg["dtVecEuler"], dtEuler)
    end
    if  comm.isMaster && reg["iTimeStep"] > 10
        save_timings(comm, reg)
    end
    print("Time step wall clock time = $(fmt_time(dtVec)) s\n")
    print("Total wall clock time excl. initialization = \
        $(fmt_time(reg["tAllTotal"])) s\n"
    )

    if comm.isMaster
        print("Evolve cloud\n")
        for chunk in chunks
            increment_time!(chunk, Δt, comm, executor)
        end
    end

    evolve!(control, executor)

    # Enable GC
    GC.enable(true)
    reg["tEulerStart"] = time()
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

# Load case setup
include("setup.jl")

if comm.size == 1
    # Serial run
    decomposition = (1, 1, 1)
end

reg["noGcTimeStepInterval"] = 100
reg["timeStepsSinceLastGC"] = 1

@show nParticles μᶜ ρᶜ nCellsPerDirection nCells origin ending decomposition
@show reg["noGcTimeStepInterval"]

mesh = Mesh(nCellsPerDirection, origin, ending, decomposition)
tNow = timing(tNow, "Initialized mesh")
control = Control(
    Locks(comm.size), Events(), ConstExtrapolator(prod(mesh.partitionN))
)
tNow = timing(tNow, "Initialized control")

# Allocate Eulerian fields
reg["eulerian"] = Vector{TwoWayEulerian{VectorField}}(undef, comm.size)
tNow = timing(tNow, "Allocated Eulerian ranks array")

# Global timers
reg["iTimeStep"] = 0
reg["nSkipInitTimeStepsProfiling"] = 2

reg["tAllStart"] = time()
reg["tAllTotal"] = 0.0
reg["dtVecAll"] = scalar[]

reg["tEulerStart"] = time()
reg["tEulerTotal"] = 0.0
reg["dtVecEuler"] = scalar[]

reg["tEvolveStart"] = time()
reg["tEvolveTotal"] = 0.0
reg["dtVecEvolve"] = scalar[]

reg["tDeviceComputeTotal"] = 0.0
reg["dtVecDeviceCompute"] = scalar[]

# Lock own Eulerian to put async evolve on wait until it's notified by call to
# evolve_cloud from OF
lock(control.locks.eulerianComms[comm.jlRank])

if comm.isHost
    nParticlesPerChunk, remainder =
        divrem(nParticles, comm.member.hostCommSize*nChunksPerDevice)
    if remainder != 0
        println(
            "Total number of particles cannot be evenly split into chunks.",
            " Thus, nParticlesPerChunk = ", nParticlesPerChunk,
            " resulting in nParticles = ", nParticles
        )
    end
    @show nParticlesPerChunk

    chunks = Vector{Chunk}(undef, nChunksPerDevice)
    # init!(chunks, mesh, nParticlesPerChunk, executor)
    allocate!(chunks, comm, executor, nParticlesPerChunk, μᶜ, ρᵈ, ρᵈ/ρᶜ)
    initWithHilbert!(chunks, mesh, comm, nParticlesPerChunk, executor)

    tNow = timing(tNow, "Initialized particle chunks")

    # Write initial state
    # write(chunks, comm, executor)
    # tNow = timing(tNow, "Written VTK data")

    # Allocate Eulerian for all ranks on master
    for i in eachindex(reg["eulerian"])
        reg["eulerian"][i] = TwoWayEulerian{VectorField}(prod(mesh.partitionN))
    end

    # Synchronize before starting the evolve thread
    MPI.Barrier(comm.communicator)

    errormonitor(
        @spawn init_async_evolve!(
            chunks, reg["eulerian"], mesh, control, comm, executor
        )
    )
    tNow = timing(tNow, "Initialized asynchronous evolve")
else
    # Allocate Eulerian only for self on slaves
    reg["eulerian"][comm.jlRank] = TwoWayEulerian{VectorField}(
        prod(mesh.partitionN)
    )

    # Synchronize before starting the evolve thread
    MPI.Barrier(comm.communicator)

    errormonitor(
        @spawn init_async_evolve!(reg["eulerian"], control, comm, executor)
    )
end

# Test few time steps when running in REPL
if isinteractive()
    # Create random velocity field
    for eulerian in reg["eulerian"]
        init_random!(eulerian.U, 2SCL, -1SCL)
    end
    tNow = timing(tNow, "Initialized velocity field")

    evolve_cloud(1e-3)
    evolve_cloud(1e-3)
end

# Run finalization function at Julia exit to let the device finish the kernels
atexit(finalize_evolve)

gcDiff = Base.GC_Diff(Base.gc_num(), reg["gc_num"])
reg["gc_num"] = Base.gc_num()
println("Julia allocations during startup: $(gcDiff.allocd/1e6) MB")

println("Load asyncParallelTracking done\n")
flush(stdout)
flush(stderr)

###############################################################################
