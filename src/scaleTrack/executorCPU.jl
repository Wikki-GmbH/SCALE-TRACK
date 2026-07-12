#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2026 Sergey Lesnik
=#

# CPU executor.  The tracking master evolves the chunks with the default-pool
# threads and works on host-side compute copies of the Eulerian fields, which
# take the role of the GPU's device copies: they decouple the tracking from
# OpenFOAM and MPI, which concurrently mutate the zero-copy shared fields.
#
# How many tracking threads a run can use follows from who else needs a core.
# The asynchronous coupling keeps one thread driving OpenFOAM, so it wants at
# most one fewer than the physical cores; the synchronous driver blocks its
# caller and can take them all.  Under MPI the ranks have to be launched
# unbound, or every thread of a rank is folded onto the one core the rank was
# pinned to.

# The CPU executor acts as a single tracking device per node: one rank per
# node becomes the tracking master and the remaining ranks only serve their
# Eulerian partitions
function count_devices_per_node(::CPU)
    return 1
end

set_device!(deviceNumber, ::CPU) = nothing

function allocate_chunk(::CPU, constructorArgs...)
    return Chunk{Vector{scalar}, Vector{Time}}(constructorArgs...)
end

function set_time!(chunk, t, Δt, ::CPU)
    chunk.time[1] = Time(t, Δt)
end

function increment_time!(chunk, Δt, executor::CPU)
    set_time!(chunk, chunk.time[1].t + Δt, Δt, executor)
end

function default_rng(::CPU)
    return Random.default_rng()
end

@inline function update!(
    eulerian::TwoWayEulerian, state0, velocity, posI, mᵈByρᶜ, ::CPU
)
    # Accumulate (do not overwrite): several parcels may reside in the same
    # cell.  Matches the GPU variant, which accumulates via atomic_add!.
    @inbounds eulerian.UTrans[posI] += mᵈByρᶜ*(state0 - velocity)
    return set_parcel_state(eulerian, velocity)
end

# On the CPU the bounding box is computed in a separate pass over the position
# arrays (compute_bounding_box!) after the evolve; the per-particle update
# used by the GPU kernel is a no-op
@inline function update!(boundingBox::BoundingBox, pos, ::CPU)
    return nothing
end

# Compute the axis-aligned bounding box of a chunk with a pass over the
# position arrays.  Called after the evolve, when the positions are final.
function compute_bounding_box!(chunk, ::CPU)
    bb = chunk.boundingBox
    bb.min[1], bb.max[1] = extrema(chunk.X)
    bb.min[2], bb.max[2] = extrema(chunk.Y)
    bb.min[3], bb.max[3] = extrema(chunk.Z)
    return nothing
end

# Evolve the particles of a chunk on the CPU.  The particle range is split
# among the default-pool threads; task tid reads the shared compute U and
# accumulates the momentum sources into its private buffers taskEulerian[tid]
# (one TwoWayEulerian per Lagrangian partition), which the caller reduces
# after all chunks are evolved.  The buffers are reused between the calls,
# keeping the hot loop allocation-free.
#
# On the first call taskThreadIds may be passed to record which OS thread
# each task lands on (see the thread-pool check in sync_evolve!).
function evolve_chunk!(
    chunk, taskEulerian, mesh, executor::CPU; taskThreadIds=nothing
)
    ΔtP = chunk.time[1].Δt / nTrackingSubSteps
    nTasks = length(taskEulerian)

    if nTasks == 1
        @inbounds for i = 1LBL:chunk.N
            evolve_particle!(
                chunk, taskEulerian[1], i, ΔtP, mesh, nTrackingSubSteps,
                executor
            )
        end
        return nothing
    end

    @sync for tid in 1:nTasks
        Threads.@spawn :default begin
            if !isnothing(taskThreadIds)
                taskThreadIds[tid] = Threads.threadid()
            end
            eulerianArr = taskEulerian[tid]
            iBegin = div(chunk.N*(tid - 1), nTasks) + 1
            iEnd = div(chunk.N*tid, nTasks)
            @inbounds for i = iBegin:iEnd
                evolve_particle!(
                    chunk, eulerianArr, i, ΔtP, mesh, nTrackingSubSteps,
                    executor
                )
            end
        end
    end
    return nothing
end

###############################################################################
# Hooks of the asynchronous tracking master

# Host-side compute copies of the Eulerian fields plus the task-private
# source buffers of the threaded evolve
struct CPUMasterState
    compute::Vector{TwoWayEulerian{VectorField}}
    task::Vector{Vector{TwoWayEulerian{VectorField}}}
end

function master_state(chunks, eulerian, mesh, comm, ::CPU)
    nTasks = Threads.nthreads(:default)

    # Host copies that own the Eulerian fields during the tracking
    compute = Vector{TwoWayEulerian{VectorField}}(undef, comm.size)
    for i in eachindex(compute)
        compute[i] = TwoWayEulerian{VectorField}(eulerian[i].N)
    end

    # Task-private views: the compute U is shared for reading, the UTrans
    # accumulator is private per task and rank to avoid write races on
    # cells shared between the tasks
    task = [
        [
            TwoWayEulerian{VectorField}(e.N, e.U, VectorField(undef, e.N))
            for e in compute
        ]
        for _ in 1:nTasks
    ]
    println(
        "CPU tracking master: ", length(chunks), " chunks evolved by ",
        nTasks, " default-pool tasks"
    )
    state = CPUMasterState(compute, task)
    reg["masterState"] = state
    return state
end

function reset_sources!(state::CPUMasterState, ::CPU)
    for ce in state.compute
        fill!(ce.UTrans, ScalarVec(0SCL, 0SCL, 0SCL))
    end
    for te in state.task, e in te
        fill!(e.UTrans, ScalarVec(0SCL, 0SCL, 0SCL))
    end
    return nothing
end

function init_bounding_boxes!(chunks, state, mesh, control, comm, executor::CPU)
    for chunk in chunks
        compute_bounding_box!(chunk, executor)
        determine!(
            comm.member.requiredEulerianRanks,
            chunk.boundingBox,
            mesh,
            control
        )
    end
    return nothing
end

function evolve_all_chunks!(chunks, state, mesh, control, comm, executor::CPU)
    for chunk in chunks
        evolve_chunk!(chunk, state.task, mesh, executor)
        compute_bounding_box!(chunk, executor)
        determine!(
            comm.member.requiredEulerianRanks,
            chunk.boundingBox,
            mesh,
            control
        )
    end
    # Reduce the task-private sources into the compute copies
    for te in state.task
        for r in eachindex(state.compute)
            UTrans = state.compute[r].UTrans
            buf = te[r].UTrans
            @inbounds @simd for i in eachindex(UTrans, buf)
                UTrans[i] += buf[i]
            end
        end
    end
    return nothing
end

###############################################################################
# Synchronous (blocking) evolve used by the sync driver

# Evolve a chunk synchronously.  With a single task the particles write their
# sources directly into the shared Eulerian field; with several tasks each
# task accumulates into a private buffer, which are reduced afterwards.  The
# buffers are allocated once and reused, keeping the hot loop allocation-free.
function sync_evolve!(chunk, eulerian, mesh, Δt, executor::CPU)
    increment_time!(chunk, Δt, executor)

    nTasks = Threads.nthreads(:default)

    if nTasks == 1
        if !haskey(reg, "syncTaskEulerian")
            reg["syncTaskEulerian"] = [[eulerian]]
        end
        evolve_chunk!(chunk, reg["syncTaskEulerian"], mesh, executor)
        return nothing
    end

    firstCall =
        !haskey(reg, "syncTaskEulerian") ||
        length(reg["syncTaskEulerian"]) != nTasks
    if firstCall
        # Read the shared velocity field, write to the private source
        reg["syncTaskEulerian"] = [
            [
                TwoWayEulerian{VectorField}(
                    eulerian.N, eulerian.U, VectorField(undef, eulerian.N)
                )
            ]
            for _ in 1:nTasks
        ]
    end
    taskEulerian::Vector{Vector{TwoWayEulerian{VectorField}}} =
        reg["syncTaskEulerian"]
    for te in taskEulerian
        fill!(te[1].UTrans, ScalarVec(0SCL, 0SCL, 0SCL))
    end

    # On the first call record which OS thread each task lands on: the tasks
    # are spawned to the :default pool and must never run on the interactive
    # thread that drives OpenFOAM (the caller of evolve_cloud)
    taskThreadIds = firstCall ? zeros(Int, nTasks) : nothing

    evolve_chunk!(chunk, taskEulerian, mesh, executor; taskThreadIds)

    if !isnothing(taskThreadIds)
        callerId = Threads.threadid()
        println(
            "Threaded tracking: caller on thread ", callerId, " (pool :",
            Threads.threadpool(), "), ", nTasks,
            " tracking tasks on threads ", taskThreadIds, " (pool :default)"
        )
        # When called from the interactive thread (the embedded solver always
        # runs with --threads=N,1) no tracking task may land on it.  Without
        # an interactive thread (standalone --threads=N) the caller's thread
        # is part of the default pool and may legitimately run a task while
        # the caller blocks in @sync.
        if Threads.threadpool() == :interactive && callerId in taskThreadIds
            error(
                "Tracking task scheduled on the thread driving OpenFOAM"
            )
        end
    end

    # Reduce the task-private sources into the shared Eulerian field
    UTrans = eulerian.UTrans
    for te in taskEulerian
        buf = te[1].UTrans
        @inbounds @simd for i in eachindex(UTrans, buf)
            UTrans[i] += buf[i]
        end
    end
    return nothing
end
