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

###############################################################################
# Auxillary Methods

function timing(t, s)
    dt = round(time() - t, sigdigits=4)
    println(s, " in ", dt, " s")

    # Invoke flush to ensure immediate printing even when executed within C code
    flush(stdout)
    return time()
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
    partitionN = N ./ decomposition
    partitionNxTimesNy = partitionN[1]*partitionN[2]
    Mesh(
        N, origin, ending, L, Δ, rΔ, decomposition, partitionN,
        partitionNxTimesNy
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
    EulerianComm::Vector{ReentrantLock}
    Chunk::ReentrantLock
end

function Locks(n)
    eulerianComm = Vector{ReentrantLock}(undef, n)
    @inbounds for i in eachindex(eulerianComm)
        eulerianComm[i] = ReentrantLock()
    end
    Locks(eulerianComm, ReentrantLock())
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

# Subscripts: c - carrier phase; d - dispersed phase
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
abstract type CommRole end
struct Master <: CommRole end
struct Slave <: CommRole end

struct Comm{T<:CommRole}
    role::T
    communicator::MPI.Comm
    isMaster::Bool
    rank::Integer
    jlRank::Integer
    masterRank::Integer
    size::Integer
end

function Comm(role, communicator)
    isMaster = (typeof(role) == Master)
    if MPI.Initialized()
        rank = MPI.Comm_rank(communicator)
        return Comm(
            role,
            communicator,
            isMaster,
            rank,
            rank + 1,
            0,
            MPI.Comm_size(communicator)
        )
    else
        rank = 0
        return Comm(role, communicator, isMaster, rank, rank + 1, 0, 1)
    end
end

# Structs used for function tagging to identify on which backend the code is
# executed
struct CPU end
struct GPU end

###############################################################################
# Methods
function initComm()
    if MPI.Initialized() && MPI.JULIA_TYPE_PTR_ATTR[]==0
        MPI.run_init_hooks()
    else
        # Construct communication with a single rank, if MPI is not initialized
        return Comm(Master(), MPI.COMM_WORLD)
    end

    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    if rank == 0
        comm = Comm(Master(), MPI.COMM_WORLD)
        MPI.versioninfo()
    else
        comm = Comm(Slave(), MPI.COMM_WORLD)
        # Suppress output from the slave ranks
        redirect_stdout(devnull)
    end
    return comm
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

function allocate_chunk(::Comm{<:CommRole}, executor, constructorArgs...) end

function set_time!(chunk, t, Δt, ::CPU)
    chunk.time[1] = Time(t, Δt)
end

function set_time!(chunk, t, Δt, ::GPU)
    CUDA.@allowscalar chunk.time[1] = Time(t, Δt)
end

function increment_time!(chunk, Δt, ::Comm{Master}, executor)
    increment_time!(chunk, Δt, executor)
end

function increment_time!(chunk, Δt, ::Comm{<:CommRole}, executor) end

function increment_time!(chunk, Δt, executor::CPU)
    set_time!(chunk, chunk.time[1].t + Δt, Δt, executor)
end

function increment_time!(chunk, Δt, executor::GPU)
    CUDA.@allowscalar t = chunk.time[1].t
    set_time!(chunk, t + Δt, Δt, executor)
end

function init!(chunk, mesh, executor, randSeed=19891)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)

    rng = Random.default_rng()
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
            copyto!(getfield(a, n), getfield(b, n))
        end
    end
    return nothing
end

# Compute cell index given position and the corresponding indices along each
# direction
@inline function locate(x, y, z, mesh)
    # TODO unsafe_trunc instead of floor is 20% faster but then numbers within
    # -1 < n < 0 are truncated to 0 which leads to incorrect localization of
    # the parcel
    iGlobal = floor(label, (x - mesh.origin.x)*mesh.rΔ.x)
    jGlobal = floor(label, (y - mesh.origin.y)*mesh.rΔ.y)
    kGlobal = floor(label, (z - mesh.origin.z)*mesh.rΔ.z)
    iPartition, iLocal = divrem(iGlobal, mesh.partitionN.x)
    jPartition, jLocal = divrem(jGlobal, mesh.partitionN.y)
    kPartition, kLocal = divrem(kGlobal, mesh.partitionN.z)

    partitionI = (
        abs(kPartition)*mesh.decomposition.x*mesh.decomposition.y
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
    for i in eachindex(control.extrapolator.prevUTrans)
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
# box is of the chunk is computed using atomics and global memory.
@inline function update!(boundingBox, pos, iParticle, ::GPU)
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

        # Reset bounding box in global memory using single thread
        if iParticle == 1
            for i in eachindex(boundingBox.min)
                boundingBox.min[i] = scalar(Inf)
                boundingBox.max[i] = scalar(-Inf)
            end
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

# Nothing to init on CPU
function init_evolve!(chunk, eulerian, mesh, control, executor::CPU)
end

# Evolve particles using CPU
function evolve!(chunk, eulerian, mesh, Δt, control, executor::CPU)
    @inbounds begin
        nSteps = 10LBL
        ΔtP = chunk.time[1].Δt / nSteps
        for i = 1LBL:chunk.N
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

function init_evolve!(chunks, eulerian, mesh, control, comm, executor::GPU)
    errormonitor(
        @spawn init_async_evolve!(
            chunks, eulerian, mesh, control, comm, executor
        )
    )
end

function init_async_evolve!(
    chunks, eulerian, mesh, control, comm::Comm{Slave}, executor::GPU
)
    # The infinite loop to be run inside an asynchronous task that is
    # specifically yielded at "lock" and "wait"
    while true
        lock(control.locks.EulerianComm[comm.jlRank]) do
            sreq = MPI.Isend(
                eulerian[comm.jlRank].U, comm.communicator,
                dest=comm.masterRank, tag=0
            )
            wait(sreq)
        end
        notify(control.events.U_copied)
        wait(control.events.Eulerian_computed)
        lock(control.locks.EulerianComm[comm.jlRank]) do
            rreq = MPI.Irecv!(
                eulerian[comm.jlRank].UTrans, comm.communicator,
                source=comm.masterRank, tag=1
            )
            wait(rreq)
            estimate_source!(eulerian[comm.jlRank].UTrans, control.extrapolator)
        end
        notify(control.events.S_copied)
    end
    return nothing
end

# CUDA call with a setup that efficiently maps to the specific GPU
function init_async_evolve!(
    chunks, eulerian, mesh, control, comm::Comm{Master}, executor::GPU
)
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

    aChunk = first(chunks)
    kernel = @cuda launch=false evolve_on_device!(
        aChunk, reg["deviceEulerianPointer"], mesh, executor
    )
    config = launch_configuration(kernel.fun)
    threads = min(aChunk.N, config.threads)
    blocks = cld(aChunk.N, threads)

    # The infinite loop to be run inside an asynchronous task that is
    # specifically yielded at "lock" and "wait"
    while true
        rreq = Vector{MPI.AbstractRequest}(undef, comm.size)
        for i in 2:comm.size
            lock(control.locks.EulerianComm[i])
            rreq[i] = MPI.Irecv!(
                eulerian[i].U, comm.communicator, source=i-1, tag=0
            )
        end

        # Reset sources on the device to zero
        for de in reg["deviceEulerian"]
            fill!(de.UTrans, ScalarVec(0SCL, 0SCL, 0SCL))  #!!
        end

        for i in 2:comm.size
            wait(rreq[i])
            unlock(control.locks.EulerianComm[i])
        end

        for i in eachindex(eulerian)
            lock(control.locks.EulerianComm[i]) do
                copyto!(reg["deviceEulerian"][i].U, eulerian[i].U)
            end
        end

        notify(control.events.U_copied)

        println("Evolve particles"); flush(stdout)
        for l in control.locks.EulerianComm
            lock(l)
        end
        lock(control.locks.Chunk) do
            for chunk in chunks
                kernel(
                    chunk, reg["deviceEulerianPointer"], mesh, executor;
                    threads, blocks
                )
            end
            CUDA.synchronize()
        end
        for l in control.locks.EulerianComm
            unlock(l)
        end
        tEvolve = time() - tStart
        global totalTime += tEvolve
        println("Lagrangian solver: waiting time for evolve to finish = \
            $(round(tEvolve, sigdigits=4)) s; total waiting time = \
            $(round(totalTime, sigdigits=4)) s"
        )
        wait(control.events.Eulerian_computed)

        # Correct and estimate the source
        lock(control.locks.EulerianComm[comm.jlRank]) do
            copyto!(
                eulerian[comm.jlRank].UTrans,
                reg["deviceEulerian"][comm.jlRank].UTrans
            )
            estimate_source!(eulerian[comm.jlRank].UTrans, control.extrapolator)
        end

        notify(control.events.S_copied)

        # Copy sources to CPU and send them to slaves
        sreq = Vector{MPI.AbstractRequest}(undef, comm.size)
        for i in 2:comm.size
            lock(control.locks.EulerianComm[i])
            copyto!(eulerian[i].UTrans, reg["deviceEulerian"][i].UTrans)
            sreq[i] = MPI.Isend(
                eulerian[i].UTrans, comm.communicator, dest=i-1, tag=1
            )
        end
        for i in 2:comm.size
            wait(sreq[i])
            unlock(control.locks.EulerianComm[i])
        end
    end
    return nothing
end

# Drives evolve on GPU
function evolve!(chunk, eulerian, mesh, Δt, control, executor::GPU)
    unlock(control.locks.EulerianComm[comm.jlRank])
    if !firstPass
        notify(control.events.Eulerian_computed)
        wait(control.events.S_copied)
    end
    global firstPass = false
    wait(control.events.U_copied)
    lock(control.locks.EulerianComm[comm.jlRank])
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
        update!(chunk.boundingBox, posNew, i, executor)
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
    lock(control.locks.Chunk) do
        copy!(reg["hostChunk"], c)
    end
    write(reg["hostChunk"], CPU())
    return nothing
end

function write(chunks, ::Comm{Master}, executor)
    GC.enable(false)
    write(chunk, executor)
    GC.enable(true)
end

function write(chunk, ::Comm{<:CommRole}, executor) end

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

function write_paraview_collection(::Comm{<:CommRole}) end


###############################################################################
# Coupling to C++

function allocate_array_j(
        name::Cstring, size::Cint, nComponents::Cint, typeByteSize::Cint
    )::Ptr{Cdouble}
    GC.enable(false)

    GC.@preserve name
    s = unsafe_string(pointer(name))

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
                    " is different from field's ", s, " size: ", Int(size)
                )
            )
        )
    end

    # By now, assume that the requested field already has been allocated
    field = getfield(reg["eulerian"][comm.jlRank], Symbol(s))
    # nComponents needs to be Int
    # n = Int(nComponents)
    # if (n == 1)
    #     field = Vector{scalar}(undef, size)
    # else
    #     field = Vector{SVector{n, scalar}}(undef, size)
    # end

    GC.enable(true)
    return Base.unsafe_convert(Ptr{Cdouble}, field)
end

const allocate_array_ptr =
    @cfunction(allocate_array_j, Ptr{Cdouble}, (Cstring, Cint, Cint, Cint))

function evolve_cloud(Δt)
    # Workaround for the segmentation fault error when using multiple Julia
    # threads.  The error occurs when the garbage collector (GC) is triggered
    # in more than one thread at the same time.  Thus, disable GC in the
    # affected functions.
    GC.enable(false)

    println("Evolve cloud")
    for chunk in chunks
        increment_time!(chunk, Δt, comm, executor)
    end
    global tStart = time()
    evolve!(chunks, reg["eulerian"], mesh, Δt, control, executor)

    # Enable GC
    GC.enable(true)
    return nothing
end

const evolve_cloud_ptr = @cfunction(evolve_cloud, Cvoid, (Cdouble,))

###############################################################################
# Global data init
comm = initComm()

println("Load asyncParallelTracking")
println("Julia active project: ", Base.active_project())

nParticles = 800_000_000  # RTX3090
# nParticles = 25_000_000  # RTX3090 fast
# nParticles = 1_000_000  # RTX3090 faster
# nParticles = 2  # GT710
nParticles = 100_000  # GT710

nChunks = 10
# nChunks = 1

executor = GPU()
# executor = CPU()

μᶜ = 1e-3
ρᶜ = 1e3
ρᵈ = 1.0
# These properties lead to large velocity source terms
# μᶜ = 1e3
# ρᵈ = 1e8
nCellsPerDirection = 40
nCells = nCellsPerDirection^3
origin = 0.0
ending = 1.0

if comm.size == 1
    # Serial run
    decomposition = (1, 1, 1)
else
    # Decomposition coefficients need to be the same as in the decomposeParDict
    decomposition = (2, 2, 1)
end

@show nParticles μᶜ ρᶜ nCellsPerDirection nCells origin ending decomposition

reg = Dict()

mesh = Mesh(nCellsPerDirection, origin, ending, decomposition)
tNow = time()
chunks = Vector{Chunk}(undef, nChunks)
if comm.isMaster
    for i in eachindex(chunks)
        chunks[i] = allocate_chunk(comm, executor, nParticles, μᶜ, ρᵈ, ρᵈ/ρᶜ)
        init!(chunks[i], mesh, executor, i)
    end
else
    chunks = []  # Placeholder for slave ranks
end
tNow = timing(tNow, "Initialized particle chunk")

# Write initial state
# write(chunk, comm, executor)
tNow = timing(tNow, "Written VTK data")

# Allocate Eulerian fields
reg["eulerian"] = Vector{TwoWayEulerian{VectorField}}(undef, comm.size)
if comm.isMaster
    @inbounds for i in eachindex(reg["eulerian"])
        reg["eulerian"][i] = TwoWayEulerian{VectorField}(prod(mesh.partitionN))
    end
else
    reg["eulerian"][comm.jlRank] = TwoWayEulerian{VectorField}(
        prod(mesh.partitionN)
    )
end

# Create random velocity field when running in REPL
if isinteractive()
    for eulerian in reg["eulerian"]
        init_random!(eulerian.U, 2SCL, -1SCL)
    end
    tNow = timing(tNow, "Initialized velocity field")
end

totalTime = 0.0
firstPass = true

control = Control(
    Locks(comm.size), Events(), ConstExtrapolator(prod(mesh.partitionN))
)
lock(control.locks.EulerianComm[comm.jlRank])
tStart = time()
init_evolve!(chunks, reg["eulerian"], mesh, control, comm, executor)
tNow = timing(tNow, "Initialized asynchronous evolve")

println("Load asyncParallelTracking done\n")
flush(stdout)
flush(stderr)

# Test few time steps when running in REPL
if isinteractive()
    evolve_cloud(1e-3)
    evolve_cloud(1e-3)
end

###############################################################################
