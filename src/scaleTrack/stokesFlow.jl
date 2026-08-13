#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    StokesFlow physics model: momentum-only two-way coupling with implicit
    Euler integration of Stokes drag.  The physics of the icoJuliaParcelFoam
    solver family.

    Everything a physics model contributes is dispatched on the model type:
    its Eulerian field set, the per-parcel arrays it needs, the mapping
    between those arrays and the state a sub-step advances, the carrier state
    read at the parcel's cell, the sub-step physics with its source
    accumulator, and the source extrapolator it defaults to.
=#

struct StokesFlow
    μᶜ::scalar      # dynamic viscosity (continuous phase)
    ρᵈ::scalar      # density (disperse phase)
    ρᵈByρᶜ::scalar  # density ratio disperse/continuous
end

StokesFlow(; μᶜ, ρᶜ, ρᵈ) = StokesFlow(μᶜ, ρᵈ, ρᵈ/ρᶜ)

# The Eulerian fields of one partition coupled in both directions: the carrier
# velocity U is read by the tracking, the momentum source UTrans is written
# back to the carrier phase
struct TwoWayEulerian{T}
    N::label
    U::T
    UTrans::T
end

function TwoWayEulerian{T}(N) where {T}
    TwoWayEulerian{T}(
        N,
        T(undef, N),
        T(undef, N)
    )
end
Adapt.@adapt_structure TwoWayEulerian

carrier_fields(::Type{<:TwoWayEulerian}) = (:U,)
source_fields(::Type{<:TwoWayEulerian}) = (:UTrans,)

host_eulerian_type(::StokesFlow) = TwoWayEulerian{VectorField}
device_eulerian_type(::StokesFlow, ex::GPU) =
    TwoWayEulerian{device_vector_type(ex, ScalarVec)}
device_eulerian_ptr_type(::StokesFlow, ex::GPU) =
    TwoWayEulerian{device_ptr_vector_type(ex, ScalarVec)}

# No model-specific per-particle arrays
parcel_props(::StokesFlow, ::Type{T}, N) where {T} = (;)

default_extrapolator(model::StokesFlow, N) =
    ConstExtrapolator(host_eulerian_type(model), N)

# Disperse-phase density, for the linear momentum of the cloud summary
parcel_density(model::StokesFlow) = model.ρᵈ

@inline function load_parcel(model::StokesFlow, c, i)
    @inbounds begin
        ⌀ = c.d[i]
        ParcelState(
            ScalarVec(c.X[i], c.Y[i], c.Z[i]),
            ScalarVec(c.U[i], c.V[i], c.W[i]),
            (⌀ = ⌀, mᵈByρᶜ = model.ρᵈByρᶜ*π*⌀^3/6SCL)
        )
    end
end

@inline function store_parcel!(::StokesFlow, c, i, parcel)
    @inbounds begin
        c.X[i] = parcel.pos.x
        c.Y[i] = parcel.pos.y
        c.Z[i] = parcel.pos.z
        c.U[i] = parcel.vel.x
        c.V[i] = parcel.vel.y
        c.W[i] = parcel.vel.z
    end
    return nothing
end

@inline function load_carrier(::StokesFlow, eulerian, posI)
    @inbounds (U = eulerian.U[posI],)
end

@inline reload_carrier(model::StokesFlow, eulerian, posI) =
    load_carrier(model, eulerian, posI)

# The accumulator holds the parcel state at the entry into the current cell
# (state0): the momentum source of a cell visit is computed from the state
# difference between cell entry and exit
@inline function init_sources(::StokesFlow, eulerian, parcel)
    return ScalarVec(parcel.vel)
end

# Implicit Euler time integration with Stokes drag
@inline function substep(model::StokesFlow, parcel, carrier, state0, Δt)
    props = parcel.props
    dragFactor = 18SCL*model.μᶜ*Δt/(model.ρᵈ*props.⌀^2)
    velNew = (parcel.vel .+ dragFactor .* carrier.U) ./ (1 + dragFactor)
    posNew = parcel.pos .+ velNew .* Δt
    return (ParcelState(posNew, velNew, props), state0)
end

# Apply the change in velocity to the source due to bounce at boundary
@inline function bounce_source(::StokesFlow, state0, velocity, componentI)
    @inbounds @reset state0[componentI] -= 2SCL*velocity[componentI]
    return state0
end

# Accumulate (do not overwrite): several parcels may reside in the same cell
# Returns the reset accumulator: the parcel state at the entry into the
# new cell.
@inline function flush_sources!(
    ::StokesFlow, eulerian, state0, parcel, posI, ::CPU
)
    @inbounds eulerian.UTrans[posI] +=
        parcel.props.mᵈByρᶜ*(state0 - parcel.vel)
    return ScalarVec(parcel.vel)
end

@inline function flush_sources!(
    ::StokesFlow, eulerian, state0, parcel, posI, executor::GPU
)
    @inbounds begin
        mᵈByρᶜ = parcel.props.mᵈByρᶜ
        for i=1LBL:3LBL
            atomic_add_component!(
                eulerian.UTrans, posI, i,
                mᵈByρᶜ*(state0[i] - parcel.vel[i]), executor
            )
        end
    end
    return ScalarVec(parcel.vel)
end
