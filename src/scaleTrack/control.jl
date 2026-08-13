#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Orchestration primitives for the asynchronous coupling: locks, events and
# the source-term extrapolators

# Locks and events that help to orchestrate the async tasks and threads
struct Locks
    chunkTransfers::ReentrantLock
    eulerianComms::Vector{ReentrantLock}
    eulerianRequest::ReentrantLock
end

function Locks(nEulerian)
    # A comprehension, since fill would alias one single lock into all slots
    eulerianComms = [ReentrantLock() for _ in 1:nEulerian]
    Locks(ReentrantLock(), eulerianComms, ReentrantLock())
end

#=
    The events the tracking task and the solver hand over on.

    U_copied and S_copied are raised by the tracking, Eulerian_computed by the
    solver, once per coupling step each.  U_locked belongs to the synchronous
    first step alone: it holds the tracking at the end of that step until the
    solver has taken its own Eulerian lock again, so that the tracking cannot
    start serving the second step against fields the solver has not yet
    claimed.  After that step the pipeline runs on the other three.
=#
struct Events
    U_copied::Event
    U_locked::Event
    S_copied::Event
    Eulerian_computed::Event

    Events() = new(Event(true), Event(true), Event(true), Event(true))
end

# The synchronization state the solver thread and the tracking task share,
# together with the extrapolator that stands in for the source between the
# steps the tracking produces one
struct Control{E <: AbstractExtrapolator}
    locks::Locks
    events::Events
    extrapolator::E
end

# Tags distinguishing the two coupling drivers; the active one is stored in
# the global trackingMode
struct AsyncMode end
struct SyncMode end
