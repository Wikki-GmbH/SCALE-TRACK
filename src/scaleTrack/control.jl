#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Orchestration primitives for the asynchronous coupling: locks, events and
# the source-term extrapolator

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
