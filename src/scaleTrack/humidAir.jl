#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2025-2026 Sergey Lesnik
    Copyright (C) 2025-2026 Henrik Rusche
    Copyright (C) 2025 Silvio Schmalfuß
=#

#=
    HumidAirDroplet physics model: water droplets in humid air with two-way
    coupled momentum, thermal energy and vapour mass transfer.  The physics of
    the buoyantHumidPimpleJuliaParcelFoam solver family, ported verbatim from
    run/buoyantHumidPimpleParcelFoam/freeFallCoolingEvaporation:

    - Schiller-Naumann drag and buoyant gravity, implicit Euler integration
    - convective heat transfer via a Ranz-Marshall style Nusselt correlation
    - droplet evaporation/condensation from the vapour saturation ratio
      (Arden-Buck saturation pressure, Kelvin curvature effect), selectable
      via the EV type parameter: Evaporation or NoEvaporation

    Ported as found (work in progress on the physics side): the source
    extrapolation is disabled (default_extrapolator = NoExtrapolator) and the
    initial carrier read clamps rhoV to non-negative values while the reread
    after a cell change does not.
=#

struct Evaporation end
struct NoEvaporation end

struct HumidAirDroplet{EV}
    μᶜ::scalar      # dynamic viscosity (continuous phase)
    ρᶜ::scalar      # density (continuous phase)
    ρᵈ::scalar      # density (disperse phase)
    g::ScalarVec    # gravitational acceleration
    Cₚᶜ::scalar     # heat capacity (carrier phase)
    Cₚᵈ::scalar     # heat capacity (dispersed phase)
    Dᵈᶜ::scalar     # diffusion coefficient of water vapour in air
    Mᵈ::scalar      # molar weight of water
    σᶜ::scalar      # surface tension of water in air
    RG::scalar      # gas constant
    SLH::scalar     # specific latent heat of water vaporisation
end

function HumidAirDroplet(
    evaporation; μᶜ, ρᶜ, ρᵈ, g, Cₚᶜ, Cₚᵈ, Dᵈᶜ, Mᵈ, σᶜ, RG, SLH
)
    HumidAirDroplet{typeof(evaporation)}(
        μᶜ, ρᶜ, ρᵈ, ScalarVec(g...), Cₚᶜ, Cₚᵈ, Dᵈᶜ, Mᵈ, σᶜ, RG, SLH
    )
end

# The Eulerian fields of one partition: carrier velocity U, temperature T and
# vapour density rhoV are read by the tracking; the momentum, thermal energy
# and vapour mass sources are written back to the carrier phase
struct HumidEulerian{TV, TS}
    N::label
    U::TV
    T::TS
    rhoV::TS
    UTrans::TV
    hTrans::TS
    rhoVTrans::TS
end

function HumidEulerian{TV, TS}(N) where {TV, TS}
    HumidEulerian{TV, TS}(
        N,
        TV(undef, N),
        TS(undef, N),
        TS(undef, N),
        TV(undef, N),
        TS(undef, N),
        TS(undef, N)
    )
end
Adapt.@adapt_structure HumidEulerian

carrier_fields(::Type{<:HumidEulerian}) = (:U, :T, :rhoV)
source_fields(::Type{<:HumidEulerian}) = (:UTrans, :hTrans, :rhoVTrans)

host_eulerian_type(::HumidAirDroplet) = HumidEulerian{VectorField, ScalarField}
device_eulerian_type(::HumidAirDroplet) =
    HumidEulerian{CuVector{ScalarVec}, CuVector{scalar}}
device_eulerian_ptr_type(::HumidAirDroplet) =
    HumidEulerian{CuDeviceVector{ScalarVec, 1}, CuDeviceVector{scalar, 1}}

# Per-particle droplet temperature
parcel_props(::HumidAirDroplet, ::Type{T}, N) where {T} = (T = T(undef, N),)

# The source extrapolation is disabled in the ported physics (see the header)
default_extrapolator(::HumidAirDroplet, N) = NoExtrapolator()

@inline function load_parcel(model::HumidAirDroplet, c, i)
    @inbounds begin
        ⌀ = scalar(c.d[i])
        ParcelState(
            ScalarVec(c.X[i], c.Y[i], c.Z[i]),
            ScalarVec(c.U[i], c.V[i], c.W[i]),
            (T = scalar(c.props.T[i]), ⌀ = ⌀, m = scalar(π*⌀^3*model.ρᵈ/6SCL))
        )
    end
end

@inline function store_parcel!(::HumidAirDroplet, c, i, parcel)
    @inbounds begin
        c.X[i] = parcel.pos.x
        c.Y[i] = parcel.pos.y
        c.Z[i] = parcel.pos.z
        c.U[i] = parcel.vel.x
        c.V[i] = parcel.vel.y
        c.W[i] = parcel.vel.z
        c.props.T[i] = parcel.props.T
        c.d[i] = parcel.props.⌀
    end
    return nothing
end

# The initial carrier read clamps the vapour density; the reread after a cell
# change (reload_carrier) does not — preserved as found
@inline function load_carrier(::HumidAirDroplet, eulerian, posI)
    @inbounds (
        U = eulerian.U[posI],
        T = eulerian.T[posI],
        rhoV = max(0SCL, eulerian.rhoV[posI])
    )
end

@inline function reload_carrier(::HumidAirDroplet, eulerian, posI)
    @inbounds (
        U = eulerian.U[posI],
        T = eulerian.T[posI],
        rhoV = eulerian.rhoV[posI]
    )
end

# Running source increments, flushed on cell change and at the end
@inline function init_sources(::HumidAirDroplet, eulerian, parcel)
    return (dUTrans = ScalarVec(0, 0, 0), dhTrans = 0SCL, drhoVTrans = 0SCL)
end

# Droplet mass change by evaporation/condensation over one sub-step: returns
# the new mass, the droplet temperature after the latent heat release, the
# new diameter and the mass change
@inline function mass_transfer(
    model::HumidAirDroplet{Evaporation}, Tᵈ, Tᶜ, rhoVᶜ, ⌀, mᵈ, Δt
)
    ρᵈ = model.ρᵈ
    Dᵈᶜ = model.Dᵈᶜ
    Mᵈ = model.Mᵈ
    σᶜ = model.σᶜ
    RG = model.RG
    SLH = model.SLH
    Cₚᵈ = model.Cₚᵈ

    # temperature in °C
    TᶜinC = scalar(Tᶜ-273.15SCL)

    # Water vapour saturation pressure (Arden-Buck-Equation) in Pa
    pvsat = scalar(611.21SCL*exp(
        (18.678SCL - TᶜinC/234.5SCL)*(TᶜinC/(257.14SCL + TᶜinC)))
    )

    # water vapour density at saturation in kg/m³
    rhovsat = scalar(Mᵈ*pvsat/(RG*Tᶜ))

    # air molecular density (43.04*Tref*p/(T*pref)) in mol/m³
    rhoMolAir = 43.04SCL*283.15SCL/(Tᶜ*1.01325SCL)

    # Water vapour pressure in Pa
    pv = rhoVᶜ/(rhoMolAir*Mᵈ)*1e5SCL

    # Saturation ratio of water vapour in continuous phase
    Sinf = pv/pvsat

    # Saturation ratio of water vapour at particle surface
    # (Only Kelvin/curvature effect, no Raoult/solute effect)
    Ssfc = scalar(exp(4SCL*Mᵈ*σᶜ/(RG*Tᵈ*ρᵈ*⌀)))

    # Integration over time using semi-implicit Euler
    # (min. particle mass equiv. to ⌀~1µm)
    mᵈNew = max(
        5e-16SCL,
        scalar(mᵈ + 2SCL*π*Dᵈᶜ*⌀*rhovsat*(Sinf - Ssfc)*Δt)
    )
    Δmᵈ = mᵈNew - mᵈ

    # Temperature change due to latent heat release
    Tᵈ += Δmᵈ*SLH/(0.5SCL*(mᵈ + mᵈNew)*Cₚᵈ)
    ⌀New = scalar(cbrt(6SCL*mᵈNew/(ρᵈ*π)))

    return (mᵈNew, Tᵈ, ⌀New, Δmᵈ)
end

@inline function mass_transfer(
    ::HumidAirDroplet{NoEvaporation}, Tᵈ, Tᶜ, rhoVᶜ, ⌀, mᵈ, Δt
)
    return (mᵈ, Tᵈ, ⌀, 0SCL)
end

@inline function substep(model::HumidAirDroplet, parcel, carrier, acc, Δt)
    μᶜ = model.μᶜ
    ρᶜ = model.ρᶜ
    ρᵈ = model.ρᵈ
    Cₚᶜ = model.Cₚᶜ
    Cₚᵈ = model.Cₚᵈ
    g = model.g

    ⌀ = parcel.props.⌀
    Tᵈ = parcel.props.T
    mᵈ = parcel.props.m
    uᵈ = parcel.vel
    uᶜ = carrier.U
    Tᶜ = carrier.T
    rhoVᶜ = carrier.rhoV

    ### mass ###
    mᵈNew, Tᵈ, ⌀New, Δmᵈ =
        mass_transfer(model, Tᵈ, Tᶜ, rhoVᶜ, ⌀, mᵈ, Δt)
    drhoVTrans = acc.drhoVTrans - Δmᵈ

    ### temperature ###
    # Thermal conductivity of air (temperature corrected)
    κᶜ = scalar(Tᵈ*8.9182E-5SCL)

    # surface values
    Tˢ = (2SCL*Tᵈ + Tᶜ)/3SCL
    TRatio = Tᶜ/Tˢ
    ρˢ = ρᶜ*TRatio
    μˢ = μᶜ/TRatio
    κˢ = κᶜ/TRatio
    Pr = Cₚᶜ*μˢ/κˢ

    # Slip velocity
    urel = uᶜ .- uᵈ

    # Particle Reynolds number
    Re = scalar(sqrt(sum(urel.^2))*⌀*ρˢ/μˢ)

    # Particle Nusselt number
    Nu = scalar(2SCL + 0.6SCL*sqrt(Re)*cbrt(Pr))

    # surface area of a sphere
    Asᵈ = scalar(π*⌀^2SCL)

    # Heat transfer coefficient
    htc = Nu*κˢ/⌀

    # integration coefficients
    bcp  = scalar(htc*Asᵈ/(mᵈ*Cₚᵈ))
    acp  = scalar(bcp*Tᶜ)

    # effective time step
    ΔtEff = Δt/(1SCL + bcp*Δt)

    ΔTᵈ = scalar((acp - bcp*Tᵈ)*ΔtEff)
    TᵈNew   = Tᵈ + ΔTᵈ
    dhTrans = acc.dhTrans - Cₚᵈ*(mᵈNew*TᵈNew - mᵈ*Tᵈ)

    ### velocity ###

    # drag force with Schiller-Naumann drag coefficient model
    CdRe = scalar(24SCL*(1SCL + (Re^0.687SCL)*0.15SCL))
    Fd = scalar(mᵈ*0.75SCL*μᶜ*CdRe/(ρᵈ*⌀^2SCL))

    # gravity force
    Fg = ScalarVec(mᵈ.*g.*(1SCL-ρᶜ/ρᵈ))

    # integration coefficients
    acpU  = ScalarVec(Fd.*uᶜ./mᵈ)
    ancp = ScalarVec(Fg./mᵈ)
    bcpU  = scalar(Fd./mᵈ)

    # effective time step
    ΔtEffU = Δt/(1SCL + bcpU*Δt)

    # Implicit Euler time integration of particle velocity
    Δuᵈ = ScalarVec((acpU .+ ancp .- bcpU.*uᵈ).*ΔtEffU)
    ΔuᵈNcp = ancp*Δt
    ΔuᵈCp  = Δuᵈ - ΔuᵈNcp
    uᵈNew   = uᵈ .+ Δuᵈ
    uᵈNewCp = uᵈ .+ ΔuᵈCp
    dUTrans = ScalarVec(
        acc.dUTrans.x - (mᵈNew*uᵈNewCp.x - mᵈ*uᵈ.x),
        acc.dUTrans.y - (mᵈNew*uᵈNewCp.y - mᵈ*uᵈ.y),
        acc.dUTrans.z - (mᵈNew*uᵈNewCp.z - mᵈ*uᵈ.z)
    )

    # update the position
    posNew = parcel.pos .+ uᵈNew.*Δt

    return (
        ParcelState(posNew, uᵈNew, (T = TᵈNew, ⌀ = ⌀New, m = mᵈNew)),
        (dUTrans = dUTrans, dhTrans = dhTrans, drhoVTrans = drhoVTrans)
    )
end

# The boundary bounce does not correct the sources in this model
@inline bounce_source(::HumidAirDroplet, acc, velocity, componentI) = acc

@inline function flush_sources!(
    ::HumidAirDroplet, eulerian, acc, parcel, posI, ::CPU
)
    # momentum transfer
    @inbounds eulerian.UTrans[posI] += acc.dUTrans

    # thermal energy transfer
    @inbounds eulerian.hTrans[posI] += acc.dhTrans

    # mass transfer
    @inbounds eulerian.rhoVTrans[posI] += acc.drhoVTrans
    return (dUTrans = ScalarVec(0, 0, 0), dhTrans = 0SCL, drhoVTrans = 0SCL)
end

@inline function flush_sources!(
    ::HumidAirDroplet, eulerian, acc, parcel, posI, ::GPU
)
    @inbounds begin
        # momentum transfer
        for i=1LBL:3LBL
            # Reinterpret the eulerian field at a location specified by posI
            # with an offset as scalar and get pointer to it.
            scalar_ptr = pointer(
                reinterpret(scalar, eulerian.UTrans), (posI-1LBL)*3LBL+i
            )
            CUDA.atomic_add!(
                scalar_ptr, (acc.dUTrans[i])
            )
        end

        # thermal energy transfer
        scalar_ptr = pointer(
            reinterpret(scalar, eulerian.hTrans), posI
        )
        CUDA.atomic_add!(scalar_ptr, (acc.dhTrans))

        # mass transfer
        scalar_ptr = pointer(
            reinterpret(scalar, eulerian.rhoVTrans), posI
        )
        CUDA.atomic_add!(scalar_ptr, (acc.drhoVTrans))
    end
    return (dUTrans = ScalarVec(0, 0, 0), dhTrans = 0SCL, drhoVTrans = 0SCL)
end
