#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# GPU executor.  Each rank hosting a GPU becomes a tracking master; the chunks
# are evolved by device kernels operating on device-side copies of the Eulerian
# fields.
#
# Everything here is vendor independent and dispatches on the abstract GPU; a
# backend supplies the vendor primitives it reaches the device through.

function count_devices_per_node(executor::GPU)
    return device_count(executor)
end

function allocate_chunk(executor::GPU, model, N, nSubSteps)
    T = device_vector_type(executor, scalar)
    A = device_vector_type(executor, Time)
    return Chunk{T, A}(N, nSubSteps, parcel_props(model, T, N))
end

function set_time!(chunk, t, Δt, executor::GPU)
    scalar_set!(chunk.time, 1, Time(t, Δt), executor)
end

function increment_time!(chunk, Δt, executor::GPU)
    t = scalar_get(chunk.time, 1, executor).t
    set_time!(chunk, t + Δt, Δt, executor)
end

# Use the device random number generator to enable processing on the device
# without copying to host
function default_rng(executor::GPU)
    return device_rng(executor)
end

# Find an extremum by comparing the element i of array arr with the value val
# using operation op.  Generic fallback for backends whose atomics do not
# offer min/max directly: spin on a compare-and-swap until the element holds
# the extremum.
@inline function atomic_extremum!(arr, i, val, op, executor::GPU)
    old = arr[i]
    while true
        assumed = old
        old, done =
            atomic_replace!(arr, i, assumed, op(assumed, val), executor)
        done && break
    end
    return nothing
end

@inline atomic_min!(arr, i, val, executor::GPU) =
    atomic_extremum!(arr, i, val, Base.min, executor)

@inline atomic_max!(arr, i, val, executor::GPU) =
    atomic_extremum!(arr, i, val, Base.max, executor)

# Calculation of the axis aligned bounding box mainly consists of identifying
# global minima and maxima of particles' positions.  This is done in two
# stages.  First, each block finds the extrema of particles contained within
# the block by using atomics and shared memory.  Second, the bounding box of
# the chunk is computed using atomics and global memory.
@inline function update!(boundingBox::BoundingBox, pos, executor::GPU)
    @inbounds begin
        # Allocate and initialize shared memory for min/max per block
        s = static_shared(scalar, Val(6), executor)
        sMin = view(s, 1:3)
        sMax = view(s, 4:6)
        if thread_index(executor) == 1
            for i in eachindex(sMin)
                sMin[i] = scalar(Inf)
                sMax[i] = scalar(-Inf)
            end
        end
        sync_block(executor)

        # Calculate min/max per block using atomics on shared memory
        for i in eachindex(sMin)
            atomic_min!(sMin, i, pos[i], executor)
            atomic_max!(sMax, i, pos[i], executor)
        end
        sync_block(executor)

        # Calculate global min/max using atomics on global memory
        if thread_index(executor) == 1
            for i in eachindex(boundingBox.min)
                atomic_min!(boundingBox.min, i, sMin[i], executor)
                atomic_max!(boundingBox.max, i, sMax[i], executor)
            end
        end
    end
    return nothing
end

# The global index of the particle handled by the calling thread
@inline function particle_index(executor::GPU)
    return (block_index(executor) - 1LBL) * block_dim(executor) +
           thread_index(executor)
end

# Kernel: compute the bounding box of a chunk from the current positions
function compute_bounding_box(chunk, executor::GPU)
    @inbounds begin
        c = chunk
        i = particle_index(executor)
        if (i <= c.N)
            pos = ScalarVec(c.X[i], c.Y[i], c.Z[i])
            update!(c.boundingBox, pos, executor)
        end
    end
    return nothing
end

# Kernel: evolve the particles of a chunk
function evolve_on_device!(chunk, model, eulerian, mesh, executor)
    @inbounds begin
        nSteps = chunk.nSubSteps
        Δtd = chunk.time[1].Δt / nSteps  # Dispersed-phase time step
        nParcels = chunk.N
        i = particle_index(executor)
        if (i <= nParcels)
            evolve_particle!(
                chunk, model, eulerian, i, Δtd, mesh, nSteps, executor
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
    copy_fields!(chunk.boundingBox, hostChunkBb)
    return hostChunkBb
end

###############################################################################
# Hooks of the asynchronous tracking master

# Device-side copies of the Eulerian fields plus the compiled kernels and
# their launch configuration
struct GPUMasterState{C, P, K1, K2}
    # Device containers that own the Eulerian fields
    compute::C
    # Device container for the pointers to the Eulerian containers.  It needs
    # to be stored separately since a device vector of device vectors is not
    # possible.
    devicePointers::P
    kernel::K1
    bbKernel::K2
    threads::Int
    blocks::Int
end

function master_state(chunks, model, eulerian, mesh, comm, executor::GPU)
    set_device!(comm.member.deviceNumber, executor)
    @debugCommPrintln("Hosting device $(current_device(executor))")

    E = device_eulerian_type(model, executor)
    compute = Vector{E}(undef, comm.size)
    devicePointers = device_vector_type(
        executor, device_eulerian_ptr_type(model, executor)
    )(
        undef,
        comm.size
    )
    for i in eachindex(compute)
        # Initialize Eulerian fields
        compute[i] = E(eulerian[i].N)
        # Get the pointers on the device and store them in the pointer
        # container.
        scalar_set!(devicePointers, i, device_convert(compute[i], executor),
            executor)
    end

    # Compile kernels and prepare configuration
    # Pick any chunk for compilation - only types are important
    aChunk = first(chunks)
    kernel = compile_kernel(
        evolve_on_device!,
        (aChunk, model, devicePointers, mesh, executor),
        executor
    )
    config = kernel_config(kernel, executor)
    # Int, not the label type the chunk size carries: the backends report
    # the occupancy in different integer types
    threads = Int(min(aChunk.N, config.threads))
    blocks = Int(cld(aChunk.N, threads))
    bbKernel = compile_kernel(
        compute_bounding_box, (aChunk, executor), executor
    )

    state = GPUMasterState(
        compute, devicePointers, kernel, bbKernel, threads, blocks
    )
    reg["masterState"] = state
    return state
end

function reset_sources!(state::GPUMasterState, ::GPU)
    for de in state.compute
        reset_sources!(de)
    end
    return nothing
end

function init_bounding_boxes!(
    chunks, model, state, mesh, control, comm, executor::GPU
)
    @sync begin
        for chunk in chunks
            @async begin
                hostChunkBb = reset_bounding_box!(chunk, executor)
                # Over the whole chunk, like the evolve kernel: every parcel
                # has to reach the reduction, or the box is not the chunk's
                launch_kernel!(
                    state.bbKernel, (chunk, executor),
                    state.threads, state.blocks, executor
                )
                copy_fields!(hostChunkBb, chunk.boundingBox)
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

function evolve_all_chunks!(
    chunks, model, state, mesh, control, comm, executor::GPU
)
    @sync begin
        for chunk in chunks
            @async begin
                hostChunkBb = reset_bounding_box!(chunk, executor)

                launch_kernel!(
                    state.kernel,
                    (chunk, model, state.devicePointers, mesh, executor),
                    state.threads, state.blocks, executor
                )

                copy_fields!(hostChunkBb, chunk.boundingBox)
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
# Synchronous (blocking) evolve used by the sync driver

function sync_evolve!(chunk, model, eulerian, mesh, Δt, executor::GPU)
    # The device copy is cached and reused.  Rebuild it when the Eulerian
    # container changes: each initialization allocates a fresh one, and a
    # different model brings a different field set with it.
    if !haskey(reg, "syncDeviceEulerian") ||
       get(reg, "syncDeviceEulerianSrc", nothing) !== eulerian
        de = device_eulerian_type(model, executor)(eulerian.N)
        dePointer = device_vector_type(
            executor, device_eulerian_ptr_type(model, executor)
        )(
            undef,
            1
        )
        scalar_set!(dePointer, 1, device_convert(de, executor), executor)
        reg["syncDeviceEulerian"] = de
        reg["syncDeviceEulerianPointer"] = dePointer
        reg["syncDeviceEulerianSrc"] = eulerian
    end
    deviceEulerian = reg["syncDeviceEulerian"]
    devicePointers = reg["syncDeviceEulerianPointer"]
    copy_fields!(deviceEulerian, eulerian)

    increment_time!(chunk, Δt, executor)

    kernel = compile_kernel(
        evolve_on_device!,
        (chunk, model, devicePointers, mesh, executor),
        executor
    )
    config = kernel_config(kernel, executor)
    threads = Int(min(chunk.N, config.threads))
    blocks = Int(cld(chunk.N, threads))

    launch_kernel!(
        kernel, (chunk, model, devicePointers, mesh, executor),
        threads, blocks, executor
    )
    synchronize_device(executor)

    copy_fields!(eulerian, deviceEulerian)
    return nothing
end
