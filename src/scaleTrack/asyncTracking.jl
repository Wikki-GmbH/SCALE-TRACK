#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    Asynchronous tracking orchestration.  The tracking runs in a task spawned
    at startup that synchronizes with the OpenFOAM time loop only through the
    locks and events in Control.

    Masters and slaves agree on which Eulerian partitions a master needs via
    a non-blocking consensus (NBC) built from point-to-point messages and
    MPI_Ibarrier.  Message tags: 2 - Eulerian request (inquiry), 3 -
    acknowledgement of an inquiry; the k-th carrier field of the model's
    Eulerian container travels with tag 10+k, the k-th source field with tag
    20+k, following the order in which the model lists them.

    The executor-specific pieces of the master loop -- the backend state, the
    initial bounding boxes, the source reset and the chunk evolve -- are
    provided as hooks by the executor.
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
            li[minP[1]:maxP[1], minP[2]:maxP[2], minP[3]:maxP[3]]
            .-
            1  # Rank indexing is zero-based
        )
    end
end

# Send the carrier fields of the own partition to every inquiring rank
function serve_eulerian(inquiringEulerianRanks, eulerian, comm, control)
    own = eulerian[comm.jlRank]
    cf = carrier_fields(typeof(own))
    sreqs = Vector{MPI.Request}(
        undef, length(inquiringEulerianRanks)*length(cf)
    )
    n = 0
    for inqHost in inquiringEulerianRanks
        lock(control.locks.eulerianComms[comm.jlRank]) do
            @debugCommPrintln("Send Eulerian to $inqHost")
            for (k, f) in enumerate(cf)
                sreqs[n += 1] = MPI.Isend(
                    getfield(own, f), comm.communicator, dest = inqHost,
                    tag = 10+k
                )
            end
        end
    end
    return sreqs
end

# Post the receives for the source fields sent back by every inquiring rank.
# Returns the requests and, per inquiring rank, a tuple of receive buffers
# (one per source field).
function receive_sources(inquiringEulerianRanks, comm, eulerian)
    inqRanks = inquiringEulerianRanks
    own = eulerian[comm.jlRank]
    sf = source_fields(typeof(own))
    sourceRreqs = Vector{MPI.Request}(undef, length(inqRanks)*length(sf))
    sourceBuffers = [
        map(f -> similar(getfield(own, f)), sf) for _ in 1:length(inqRanks)
    ]
    n = 0
    for (i, inqRank) in enumerate(inqRanks)
        @debugCommPrintln("Receive sources from $inqRank")
        for (k, f) in enumerate(sf)
            sourceRreqs[n += 1] = MPI.Irecv!(
                sourceBuffers[i][k], comm.communicator; source = inqRank,
                tag = 20+k
            )
        end
    end
    return (sourceRreqs, sourceBuffers)
end

# Wait for the posted source receives and accumulate the buffers into the own
# partition's source fields
function accumulate_sources!(eulerian, sourceRreqs, sourceBuffers, comm)
    own = eulerian[comm.jlRank]
    sf = source_fields(typeof(own))
    n = 0
    for buffers in sourceBuffers
        for (k, f) in enumerate(sf)
            comm_wait(sourceRreqs[n += 1])
            getfield(own, f) .+= buffers[k]
        end
    end
    return nothing
end

# Answer one pending Eulerian request, if any: receive the empty inquiry
# (tag 2), acknowledge it (tag 3) and record the inquiring rank
function probe_eulerian_inquiry!(sreqs, inqRanks, comm, control, probeFlag)
    comm_Iprobe(comm.communicator, probeFlag; tag = 2)
    if probeFlag[] != 0
        probeFlag[] = 0
        bufRef = Ref(41)
        _, status = MPI.Recv!(
            MPI.Buffer(bufRef), comm.communicator, MPI.Status; tag = 2
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
# and models
function init_async_evolve!(eulerian, control, comm::Comm{Slave}, executor)
    # Counts the steps this task has served
    iEvolve = 0

    # The infinite loop to be run inside an asynchronous task that is
    # specifically yielded at "lock" and "wait"
    while true
        iEvolve += 1

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

        for req in sreqs
            comm_wait(req)
        end
        eulerianSreqs = serve_eulerian(inqRanks, eulerian, comm, control)
        for req in eulerianSreqs
            comm_wait(req)
        end

        notify(control.events.U_copied)
        wait(control.events.Eulerian_computed)

        sourceRreqs, sourceBuffers = receive_sources(inqRanks, comm, eulerian)
        lock(control.locks.eulerianComms[comm.jlRank]) do
            # The sources of this step are the contributions about to be
            # received and nothing else, so the fields are cleared before they
            # are accumulated into.
            reset_sources!(eulerian[comm.jlRank])
            accumulate_sources!(eulerian, sourceRreqs, sourceBuffers, comm)
        end

        estimate_source!(eulerian[comm.jlRank], control.extrapolator)
        empty!(inqRanks)

        notify(control.events.S_copied)
        iEvolve == 1 && wait(control.events.U_locked)
    end
    return nothing
end

# Tracking-master loop.  The executor determines where the chunks are evolved
# and where the compute copies of the Eulerian fields live (host memory for
# the CPU, device memory for the GPU); the model determines the field set and
# the particle physics; the communication and orchestration logic is
# identical for all executors and models.
function init_async_evolve!(
    chunks, model, eulerian, mesh, control, comm::Comm{Master}, executor
)
    state = master_state(chunks, model, eulerian, mesh, comm, executor)
    E = host_eulerian_type(model)
    cf = carrier_fields(E)
    sf = source_fields(E)
    nCf = length(cf)
    nSf = length(sf)

    # Initialization: bounding boxes of chunks are required before the main
    # loop may be run
    lock(control.locks.chunkTransfers) do
        init_bounding_boxes!(
            chunks, model, state, mesh, control, comm, executor
        )
    end

    # Counts the evolves this task has completed, which trails the solver's
    # step count by whatever the coupling has in flight
    iEvolve = 0

    # The infinite loop to be run inside an asynchronous task that is
    # specifically yielded at "lock" and "wait"
    while true
        iEvolve += 1

        tNegotiate = time()

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
        rreqs = Vector{MPI.Request}(undef, nRequests*nCf)

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
            # Setup receives for the requested carrier fields
            lock(control.locks.eulerianComms[iRank₁])
            for (k, f) in enumerate(cf)
                rreqs[(i - 1) * nCf + k] = MPI.Irecv!(
                    getfield(eulerian[iRank₁], f), comm.communicator,
                    source = iRank₀, tag = 10+k
                )
            end
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

        for req in eulerianSreqs
            comm_wait(req)
        end

        for (i, _, iRank₁) in zipIter
            for k in 1:nCf
                comm_wait(rreqs[(i - 1) * nCf + k])
            end
            unlock(control.locks.eulerianComms[iRank₁])
        end

        record_timing!("negotiate", time() - tNegotiate, iEvolve)
        tCopyCarrier = time()

        # Copy all required Eulerian carrier fields to the compute copies
        allReqRanks₁ =
            Iterators.map(x -> x+1, comm.member.requiredEulerianRanks)
        for iRank₁ in allReqRanks₁
            lock(control.locks.eulerianComms[iRank₁]) do
                for f in cf
                    copyto!(
                        getfield(state.compute[iRank₁], f),
                        getfield(eulerian[iRank₁], f)
                    )
                end
            end
        end

        empty!(comm.member.requiredEulerianRanks)

        record_timing!("copyCarrier", time() - tCopyCarrier, iEvolve)

        # Nothing in this loop writes to stdout.  A redirected stdout is served
        # by libuv on the thread that runs the event loop, which is the thread
        # sitting in the Eulerian solver, so a write from here does not
        # complete until the solver returns -- one print costs a whole Eulerian
        # phase and the kernel launch waits behind it.  What the tracking has
        # to report is recorded in the timing series and reported by the
        # solver.

        notify(control.events.U_copied)

        tCompute = time()
        lock(control.locks.chunkTransfers) do
            evolve_all_chunks!(
                chunks, model, state, mesh, control, comm, executor
            )
        end
        dtCompute = time() - tCompute
        record_timing!("deviceCompute", dtCompute, iEvolve)

        tWaitEuler = time()
        wait(control.events.Eulerian_computed)
        record_timing!("waitEuler", time() - tWaitEuler, iEvolve)
        tCopySource = time()

        # Correct and estimate the source
        lock(control.locks.eulerianComms[comm.jlRank]) do
            for f in sf
                copyto!(
                    getfield(eulerian[comm.jlRank], f),
                    getfield(state.compute[comm.jlRank], f)
                )
            end
            estimate_source!(eulerian[comm.jlRank], control.extrapolator)
        end

        record_timing!("copySource", time() - tCopySource, iEvolve)
        tExchangeSource = time()

        sourceRreqs, sourceBuffers = receive_sources(inqRanks, comm, eulerian)

        # Send the sources to the inquiring ranks
        sourceSreqs = Vector{MPI.Request}(undef, nRequests*nSf)

        for (i, iRank₀, iRank₁) in zipIter
            @debugCommPrintln("Send sources to $iRank₀")
            lock(control.locks.eulerianComms[iRank₁])
            for (k, f) in enumerate(sf)
                copyto!(
                    getfield(eulerian[iRank₁], f),
                    getfield(state.compute[iRank₁], f)
                )
                sourceSreqs[(i - 1) * nSf + k] = MPI.Isend(
                    getfield(eulerian[iRank₁], f), comm.communicator,
                    dest = iRank₀, tag = 20+k
                )
            end
        end

        for (i, _, iRank₁) in zipIter
            for k in 1:nSf
                comm_wait(sourceSreqs[(i - 1) * nSf + k])
            end
            unlock(control.locks.eulerianComms[iRank₁])
        end

        lock(control.locks.eulerianComms[comm.jlRank]) do
            accumulate_sources!(eulerian, sourceRreqs, sourceBuffers, comm)
            estimate_source!(eulerian[comm.jlRank], control.extrapolator)
        end

        empty!(inqRanks)

        record_timing!("exchangeSource", time() - tExchangeSource, iEvolve)

        notify(control.events.S_copied)
        iEvolve == 1 && wait(control.events.U_locked)
    end
    return nothing
end

# Drives the asynchronous evolve from the OpenFOAM time loop: pure lock/event
# logic, shared by all executors and by masters and slaves
function evolve!(control, executor)
    tWait = time()
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
        notify(control.events.U_locked)
        record_timing!("wait", time() - tWait, iStep)
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
    record_timing!("wait", time() - tWait, iStep)
    return nothing
end
