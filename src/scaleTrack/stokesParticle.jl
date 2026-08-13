#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    StokesParticle physics model: momentum-only two-way coupling with implicit
    Euler integration of Stokes drag.  The physics of the icoJuliaParcelFoam
    solver family.

    The carrier velocity is read at the parcel's cell and never solved for --
    the continuous phase belongs to OpenFOAM.  What is modelled here is one
    particle relaxing towards it, over a relaxation time set by the particle's
    diameter and density and the carrier's viscosity.
=#

struct StokesParticle
    μᶜ::scalar      # dynamic viscosity (continuous phase)
    ρᵈ::scalar      # density (disperse phase)
    ρᵈByρᶜ::scalar  # density ratio disperse/continuous
end

StokesParticle(; μᶜ, ρᶜ, ρᵈ) = StokesParticle(μᶜ, ρᵈ, ρᵈ/ρᶜ)

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

host_eulerian_type(::StokesParticle) = TwoWayEulerian{VectorField}
device_eulerian_type(::StokesParticle, ex::GPU) =
    TwoWayEulerian{device_vector_type(ex, ScalarVec)}
device_eulerian_ptr_type(::StokesParticle, ex::GPU) =
    TwoWayEulerian{device_ptr_vector_type(ex, ScalarVec)}

# No model-specific per-particle arrays
parcel_props(::StokesParticle, ::Type{T}, N) where {T} = (;)

default_extrapolator(model::StokesParticle, N) =
    ConstExtrapolator(host_eulerian_type(model), N)

# Disperse-phase density, for the linear momentum of the cloud summary
parcel_density(model::StokesParticle) = model.ρᵈ

@inline function load_parcel(model::StokesParticle, c, i)
    @inbounds begin
        ⌀ = c.d[i]
        ParcelState(
            ScalarVec(c.X[i], c.Y[i], c.Z[i]),
            ScalarVec(c.U[i], c.V[i], c.W[i]),
            (⌀ = ⌀, mᵈByρᶜ = model.ρᵈByρᶜ*π*⌀^3/6SCL)
        )
    end
end

@inline function store_parcel!(::StokesParticle, c, i, parcel)
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

@inline function load_carrier(::StokesParticle, eulerian, posI)
    @inbounds (U = eulerian.U[posI],)
end

@inline reload_carrier(model::StokesParticle, eulerian, posI) =
    load_carrier(model, eulerian, posI)

# The accumulator holds the parcel state at the entry into the current cell
# (state0): the momentum source of a cell visit is computed from the state
# difference between cell entry and exit
@inline function init_sources(::StokesParticle, eulerian, parcel)
    return ScalarVec(parcel.vel)
end

# Implicit Euler time integration with Stokes drag
@inline function substep(model::StokesParticle, parcel, carrier, state0, Δt)
    props = parcel.props
    dragFactor = 18SCL*model.μᶜ*Δt/(model.ρᵈ*props.⌀^2)
    velNew = (parcel.vel .+ dragFactor .* carrier.U) ./ (1 + dragFactor)
    posNew = parcel.pos .+ velNew .* Δt
    return (ParcelState(posNew, velNew, props), state0)
end

# Apply the change in velocity to the source due to bounce at boundary
@inline function bounce_source(::StokesParticle, state0, velocity, componentI)
    @inbounds @reset state0[componentI] -= 2SCL*velocity[componentI]
    return state0
end

# Accumulate (do not overwrite): several parcels may reside in the same cell
# Returns the reset accumulator: the parcel state at the entry into the
# new cell.
@inline function flush_sources!(
    ::StokesParticle, eulerian, state0, parcel, posI, ::CPU
)
    @inbounds eulerian.UTrans[posI] +=
        parcel.props.mᵈByρᶜ*(state0 - parcel.vel)
    return ScalarVec(parcel.vel)
end

@inline function flush_sources!(
    ::StokesParticle, eulerian, state0, parcel, posI, executor::GPU
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
