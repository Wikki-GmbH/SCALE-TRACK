#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    Asynchronous tracking orchestration.  The tracking runs in a task spawned
    at startup (init_async_evolve!) that synchronizes with the OpenFOAM time
    loop only through the locks and events in Control; evolve!(control,
    executor) is the counterpart driven from evolve_cloud each time step.

    Masters and slaves agree on which Eulerian partitions a master needs via
    a non-blocking consensus (NBC) built from point-to-point messages and
    MPI_Ibarrier.  Message tags: 0 - Eulerian velocity field, 1 - source
    field, 2 - Eulerian request (inquiry), 3 - acknowledgement of an inquiry.

    The executor-specific pieces of the master loop are provided as hooks by
    executorCPU.jl and executorGPU.jl:

    - master_state(chunks, eulerian, mesh, comm, executor): backend state
      (compute copies of the Eulerian fields with a field .compute, kernels,
      task buffers)
    - init_bounding_boxes!(chunks, state, mesh, control, comm, executor)
    - reset_sources!(state, executor)
    - evolve_all_chunks!(chunks, state, mesh, control, comm, executor)
=#

# Comm-dispatch helpers: only tracking masters own particle chunks
function allocate_chunk(::Comm{Master}, executor, constructorArgs...)
    return allocate_chunk(executor, constructorArgs...)
end

function increment_time!(chunk, Δt, ::Comm{Master}, executor)
    increment_time!(chunk, Δt, executor)
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

# Answer one pending Eulerian request, if any: receive the empty inquiry
# (tag 2), acknowledge it (tag 3) and record the inquiring rank
function probe_eulerian_inquiry!(sreqs, inqRanks, comm, control, probeFlag)
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
    return nothing
end

# Eulerian-serving slave loop: pure MPI/event logic, shared by all executors
function init_async_evolve!(eulerian, control, comm::Comm{Slave}, executor)
    # The infinite loop to be run inside an asynchronous task that is
    # specifically yielded at "lock" and "wait"
    while true
        # Non-blocking consensus for processing Eulerian requests
        barrierFlag = Ref{Cint}(0)
        probeFlag = Ref{Cint}(0)
        # Nothing to send, hence, set the barrier directly
        breq = MPI.Ibarrier(comm.communicator)
        sreqs = Vector{MPI.Request}()
        inqRanks = comm.member.inquiringEulerianRanks
        while barrierFlag[] == 0
            probe_eulerian_inquiry!(sreqs, inqRanks, comm, control, probeFlag)
            MPI.Testall(sreqs)
            comm_test(breq, barrierFlag)
        end

        for req in sreqs comm_wait(req) end
        eulerianSreqs = serve_eulerian(inqRanks, eulerian, comm, control)
        for req in eulerianSreqs comm_wait(req) end

        notify(control.events.U_copied)
        wait(control.events.Eulerian_computed)

        sourceRreqs, sourceBuffers = receive_sources(inqRanks, comm, eulerian)
        lock(control.locks.eulerianComms[comm.jlRank]) do
            # The sources of this step are the contributions about to be
            # received and nothing else, so the fields are cleared before they
            # are accumulated into.
            fill!(eulerian[comm.jlRank].UTrans, ScalarVec(0SCL, 0SCL, 0SCL))
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

# Tracking-master loop.  The executor determines where the chunks are evolved
# and where the compute copies of the Eulerian fields live (host memory for
# the CPU, device memory for the GPU); the communication and orchestration
# logic is identical for all executors.
function init_async_evolve!(
    chunks, eulerian, mesh, control, comm::Comm{Master}, executor
)
    state = master_state(chunks, eulerian, mesh, comm, executor)

    # Initialization: bounding boxes of chunks are required before the main
    # loop may be run
    lock(control.locks.chunkTransfers) do
        init_bounding_boxes!(chunks, state, mesh, control, comm, executor)
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
            probe_eulerian_inquiry!(sreqs, inqRanks, comm, control, probeFlag)
            if barrierOn
                comm_test(breq, barrierFlag)
            elseif MPI.Testall(sreqs)
                breq = MPI.Ibarrier(comm.communicator)
                barrierOn = true
            end
        end

        eulerianSreqs = serve_eulerian(inqRanks, eulerian, comm, control)

        # Reset the sources of the compute copies (and, on the CPU, the
        # task-private buffers) to zero
        reset_sources!(state, executor)

        for req in eulerianSreqs comm_wait(req) end

        for (i, _, iRank₁) in zipIter
            comm_wait(rreqs[i])
            unlock(control.locks.eulerianComms[iRank₁])
        end

        # Copy all required Eulerian to the compute copies
        allReqRanks₁ =
            Iterators.map(x -> x+1, comm.member.requiredEulerianRanks)
        for iRank₁ in allReqRanks₁
            lock(control.locks.eulerianComms[iRank₁]) do
                copyto!(state.compute[iRank₁].U, eulerian[iRank₁].U)
            end
        end

        empty!(comm.member.requiredEulerianRanks)

        notify(control.events.U_copied)

        print("Evolve particles\n")

        lock(control.locks.chunkTransfers) do
            evolve_all_chunks!(chunks, state, mesh, control, comm, executor)
        end

        tEvolve = time() - tStart
        global totalTime += tEvolve
        print("Lagrangian solver: waiting time for evolve to finish = \
            $(round(tEvolve, sigdigits=4)) s; total waiting time = \
            $(round(totalTime, sigdigits=4)) s\n"
        )

        wait(control.events.Eulerian_computed)

        # Correct and estimate the source
        lock(control.locks.eulerianComms[comm.jlRank]) do
            copyto!(
                eulerian[comm.jlRank].UTrans,
                state.compute[comm.jlRank].UTrans
            )
            estimate_source!(eulerian[comm.jlRank].UTrans, control.extrapolator)
        end

        sourceRreqs, sourceBuffers = receive_sources(inqRanks, comm, eulerian)

        # Send the sources to the inquiring ranks
        sourceSreqs = Vector{MPI.Request}(undef, nRequests)

        for (i, iRank₀, iRank₁) in zipIter
            @debugCommPrintln("Send source to $iRank₀")
            lock(control.locks.eulerianComms[iRank₁])
            copyto!(
                eulerian[iRank₁].UTrans, state.compute[iRank₁].UTrans
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

# Drives the asynchronous evolve from the OpenFOAM time loop: pure lock/event
# logic, shared by all executors and by masters and slaves
function evolve!(control, executor)
    iStep = reg["timeStep"]
    unlock(control.locks.eulerianComms[comm.jlRank])

    if iStep == 1
        # The first step runs synchronously, so that the Eulerian phase never
        # advances on sources that no tracking has produced yet.  An
        # extrapolator started from those carries the error of its first
        # estimate into every step that follows.
        wait(control.events.U_copied)
        notify(control.events.Eulerian_computed)
        wait(control.events.S_copied)
        lock(control.locks.eulerianComms[comm.jlRank])
        return nothing
    elseif iStep > 2
        # The sources awaited here are the ones the previous step set off,
        # which is what leaves the tracking free to run through the Eulerian
        # phase.  Step 2 has none of its own to wait for: the synchronous step
        # consumed them, and skipping it here is what offsets the pipeline.
        notify(control.events.Eulerian_computed)
        wait(control.events.S_copied)
    end

    wait(control.events.U_copied)
    lock(control.locks.eulerianComms[comm.jlRank])
    return nothing
end
