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

# Extrapolation of sources
abstract type AbstractExtrapolator end

# Constant extrapolator: holds one "previous true source" array per source
# field of the Eulerian container (keyed by field name)
struct ConstExtrapolator{P} <: AbstractExtrapolator
    prev::P
end

function ConstExtrapolator(::Type{E}, N::Real) where {E}
    prev = NamedTuple{source_fields(E)}(
        map(source_fields(E)) do f
            arr = fieldtype(E, f)(undef, N)
            fill!(arr, zero(eltype(arr)))
            arr
        end
    )
    ConstExtrapolator(prev)
end

# No extrapolation: the sources are passed on unmodified
struct NoExtrapolator <: AbstractExtrapolator end

struct Control{E <: AbstractExtrapolator}
    locks::Locks
    events::Events
    extrapolator::E
end

# Constant extrapolator assumes that the current source is the same as the true
# from the previous time step.  The estimated source from the previous time
# step is corrected using the true source from the previous time step.
# estSⁿ = trueSⁿ⁻¹ - estSⁿ⁻¹ + extrapSⁿ, with extrapSⁿ = trueSⁿ⁻¹
function estimate_source!(eulerian, extrapolator::ConstExtrapolator)
    for f in keys(extrapolator.prev)
        curr = getfield(eulerian, f)
        prev = getfield(extrapolator.prev, f)
        curr .= 2.0 .* curr .- prev
        for i in eachindex(prev)
            @inbounds prev[i] = curr[i]
        end
    end
    return nothing
end

estimate_source!(eulerian, ::NoExtrapolator) = nothing

# Resolve the extrapolator a case asked for.  Accepted are nothing (use the
# model's default), a ready instance, or a type to be constructed for this
# model and partition size -- the last lets a case select an extrapolator
# without knowing the partition size.
make_extrapolator(::Nothing, model, N) = default_extrapolator(model, N)
make_extrapolator(e::AbstractExtrapolator, model, N) = e
make_extrapolator(::Type{NoExtrapolator}, model, N) = NoExtrapolator()
make_extrapolator(::Type{E}, model, N) where {E <: AbstractExtrapolator} =
    E(host_eulerian_type(model), N)
