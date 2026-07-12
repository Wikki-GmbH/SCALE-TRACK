#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# GPU (CUDA) executor.  Each rank hosting a GPU becomes a tracking master; the
# chunks are evolved by CUDA kernels operating on device-side copies of the
# Eulerian fields.

function count_devices_per_node(::GPU)
    return length(CUDA.devices())
end

set_device!(deviceNumber, ::GPU) = CUDA.device!(deviceNumber)

function allocate_chunk(::GPU, constructorArgs...)
    return Chunk{CuVector{scalar}, CuVector{Time}}(constructorArgs...)
end

function set_time!(chunk, t, Δt, ::GPU)
    CUDA.@allowscalar chunk.time[1] = Time(t, Δt)
end

function increment_time!(chunk, Δt, executor::GPU)
    CUDA.@allowscalar t = chunk.time[1].t
    set_time!(chunk, t + Δt, Δt, executor)
end

# Use CUDA random number generator to enable processing on the device without
# copying to host
function default_rng(::GPU)
    return CUDA.default_rng()
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
@inline function update!(boundingBox::BoundingBox, pos, ::GPU)
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

# Kernel: compute the bounding box of a chunk from the current positions
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

# Kernel: evolve the particles of a chunk
function evolve_on_device!(chunk, eulerian, mesh, executor)
    @inbounds begin
        Δtd = chunk.time[1].Δt / nTrackingSubSteps  # Dispersed-phase time step
        nParticles = chunk.N
        i = (blockIdx().x - 1LBL) * blockDim().x + threadIdx().x
        if(i <= nParticles)
            evolve_particle!(
                chunk, eulerian, i, Δtd, mesh, nTrackingSubSteps, executor
            )
        end
    end
    return nothing
end

# Reset the bounding box in device memory to an empty (inverted) box
function reset_bounding_box!(chunk, ::GPU)
    hostChunkBb = BoundingBox{Vector{scalar}}(
        fill(Inf, 3), fill(-Inf, 3)
    )
    copy!(chunk.boundingBox, hostChunkBb)
    return hostChunkBb
end

###############################################################################
# Hooks of the asynchronous tracking master

# Device-side copies of the Eulerian fields plus the compiled kernels and
# their launch configuration
struct GPUMasterState{C, P, K1, K2}
    # CUDA containers that own the Eulerian fields on the device
    compute::C
    # CUDA container for the pointers (CuDeviceVector) to the Eulerian
    # containers.  It needs to be stored separately since CuVector{CuVector}
    # is not possible.
    devicePointers::P
    kernel::K1
    bbKernel::K2
    threads::Int
    blocks::Int
end

function master_state(chunks, eulerian, mesh, comm, executor::GPU)
    CUDA.device!(comm.member.deviceNumber)
    @debugCommPrintln("Hosting device $(CUDA.device())")

    compute = Vector{TwoWayEulerian{CuVector{ScalarVec}}}(undef, comm.size)
    devicePointers =
        CuVector{TwoWayEulerian{CuDeviceVector{ScalarVec, 1}}}(
            undef, comm.size
        )
    for i in eachindex(compute)
        # Initialize Eulerian fields
        compute[i] = TwoWayEulerian{CuVector{ScalarVec}}(eulerian[i].N)
        # Get the pointers on the device (cudaconvert) and store them in
        # the pointer container.
        CUDA.@allowscalar devicePointers[i] = cudaconvert(compute[i])
    end

    # Compile kernels and prepare configuration
    # Pick any chunk for compilation - only types are important
    aChunk = first(chunks)
    kernel = @cuda launch=false evolve_on_device!(
        aChunk, devicePointers, mesh, executor
    )
    config = launch_configuration(kernel.fun)
    threads = min(aChunk.N, config.threads)
    blocks = cld(aChunk.N, threads)
    bbKernel = @cuda launch=false compute_bounding_box(aChunk, executor)

    state = GPUMasterState(
        compute, devicePointers, kernel, bbKernel, threads, blocks
    )
    reg["masterState"] = state
    return state
end

function reset_sources!(state::GPUMasterState, ::GPU)
    for de in state.compute
        fill!(de.UTrans, ScalarVec(0SCL, 0SCL, 0SCL))
    end
    return nothing
end

function init_bounding_boxes!(chunks, state, mesh, control, comm, executor::GPU)
    @sync begin
        for chunk in chunks
            @async begin
                hostChunkBb = reset_bounding_box!(chunk, executor)
                # Over the whole chunk, like the evolve kernel: every parcel
                # has to reach the reduction, or the box is not the chunk's
                state.bbKernel(
                    chunk, executor;
                    threads=state.threads, blocks=state.blocks
                )
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
    return nothing
end

function evolve_all_chunks!(chunks, state, mesh, control, comm, executor::GPU)
    @sync begin
        for chunk in chunks
            @async begin
                hostChunkBb = reset_bounding_box!(chunk, executor)

                state.kernel(
                    chunk, state.devicePointers, mesh, executor;
                    threads=state.threads, blocks=state.blocks
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
    return nothing
end

###############################################################################
# Synchronous (blocking) evolve used by the sync driver (see
# init_sync_tracking! in coupling.jl)

function sync_evolve!(chunk, eulerian, mesh, Δt, executor::GPU)
    if !haskey(reg, "syncDeviceEulerian")
        de = TwoWayEulerian{CuVector{ScalarVec}}(eulerian.N)
        dePointer =
            CuVector{TwoWayEulerian{CuDeviceVector{ScalarVec, 1}}}(undef, 1)
        CUDA.@allowscalar dePointer[1] = cudaconvert(de)
        reg["syncDeviceEulerian"] = de
        reg["syncDeviceEulerianPointer"] = dePointer
    end
    deviceEulerian = reg["syncDeviceEulerian"]
    devicePointers = reg["syncDeviceEulerianPointer"]
    copy!(deviceEulerian, eulerian)

    increment_time!(chunk, Δt, executor)

    kernel = @cuda launch=false evolve_on_device!(
        chunk, devicePointers, mesh, executor
    )
    config = launch_configuration(kernel.fun)
    threads = min(chunk.N, config.threads)
    blocks = cld(chunk.N, threads)

    CUDA.@sync kernel(chunk, devicePointers, mesh, executor; threads, blocks)

    copy!(eulerian, deviceEulerian)
    return nothing
end
