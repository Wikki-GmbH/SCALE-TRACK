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

function allocate_chunk(::CPU, model, N, nSubSteps)
    T = Vector{scalar}
    return Chunk{T, Vector{Time}}(N, nSubSteps, parcel_props(model, T, N))
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

# On the CPU the bounding box is computed in a separate pass after the evolve,
# so the per-particle update is a no-op here
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
# among the default-pool threads; task tid reads the shared compute carrier
# fields and accumulates the sources into its private buffers
# taskEulerian[tid] (one Eulerian container per Lagrangian partition), which
# the caller reduces after all chunks are evolved.  The buffers are reused
# between the calls, keeping the hot loop allocation-free.
#
# On the first call taskThreadIds may be passed to record which OS thread
# each task lands on.
function evolve_chunk!(
    chunk, model, taskEulerian, mesh, executor::CPU; taskThreadIds=nothing
)
    nSteps = chunk.nSubSteps
    ΔtP = chunk.time[1].Δt / nSteps
    nTasks = length(taskEulerian)

    if nTasks == 1
        @inbounds for i = 1LBL:chunk.N
            evolve_particle!(
                chunk, model, taskEulerian[1], i, ΔtP, mesh, nSteps, executor
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
                    chunk, model, eulerianArr, i, ΔtP, mesh, nSteps, executor
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
struct CPUMasterState{E}
    compute::Vector{E}
    task::Vector{Vector{E}}
end

function master_state(chunks, model, eulerian, mesh, comm, ::CPU)
    nTasks = Threads.nthreads(:default)
    E = host_eulerian_type(model)

    # Host copies that own the Eulerian fields during the tracking
    compute = Vector{E}(undef, comm.size)
    for i in eachindex(compute)
        compute[i] = E(eulerian[i].N)
    end

    # Task-private views: the compute carrier fields are shared for reading,
    # the source accumulators are private per task and rank to avoid write
    # races on cells shared between the tasks
    task = [[task_view(e) for e in compute] for _ in 1:nTasks]
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
        reset_sources!(ce)
    end
    for te in state.task, e in te
        reset_sources!(e)
    end
    return nothing
end

function init_bounding_boxes!(
    chunks, model, state, mesh, control, comm, executor::CPU
)
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

function evolve_all_chunks!(
    chunks, model, state, mesh, control, comm, executor::CPU
)
    for chunk in chunks
        evolve_chunk!(chunk, model, state.task, mesh, executor)
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
            reduce_sources!(state.compute[r], te[r])
        end
    end
    return nothing
end

# Add the source fields of a task-private view into the compute copy
function reduce_sources!(compute::E, taskView::E) where {E}
    for f in source_fields(E)
        dst = getfield(compute, f)
        src = getfield(taskView, f)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] += src[i]
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
function sync_evolve!(chunk, model, eulerian, mesh, Δt, executor::CPU)
    increment_time!(chunk, Δt, executor)

    nTasks = Threads.nthreads(:default)

    # The task buffers are cached and reused.  Rebuild them when the task
    # count changes or when the Eulerian container does -- each initialization
    # allocates a fresh one.
    firstCall =
        !haskey(reg, "syncTaskEulerian") ||
        get(reg, "syncEulerianSrc", nothing) !== eulerian ||
        length(reg["syncTaskEulerian"]) != nTasks

    if nTasks == 1
        if firstCall
            reg["syncTaskEulerian"] = [[eulerian]]
            reg["syncEulerianSrc"] = eulerian
        end
        evolve_chunk!(chunk, model, reg["syncTaskEulerian"], mesh, executor)
        return nothing
    end

    if firstCall
        # Read the shared carrier fields, write to the private sources
        reg["syncTaskEulerian"] = [[task_view(eulerian)] for _ in 1:nTasks]
        reg["syncEulerianSrc"] = eulerian
    end
    taskEulerian = reg["syncTaskEulerian"]
    for te in taskEulerian
        reset_sources!(te[1])
    end

    # On the first call record which OS thread each task lands on: the tasks
    # are spawned to the :default pool and must never run on the interactive
    # thread that drives OpenFOAM (the caller of evolve_cloud)
    taskThreadIds = firstCall ? zeros(Int, nTasks) : nothing

    evolve_chunk!(chunk, model, taskEulerian, mesh, executor; taskThreadIds)

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
    for te in taskEulerian
        reduce_sources!(eulerian, te[1])
    end
    return nothing
end
