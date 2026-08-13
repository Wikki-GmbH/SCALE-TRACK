#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Source extrapolation.  The tracking hands the carrier a source once per
# coupling step, but the solver advances every step, so between the two it
# needs an estimate.  A model names the extrapolator it defaults to; a case
# may override it.
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
