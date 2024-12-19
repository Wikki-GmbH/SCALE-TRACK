using DifferentialEquations
using Random
using Accessors

# Misc
const Δt = 1e-1
const tEnd = 3e1
const Cd = 0.47

# Carrier Phase
const Vᶜ = 1e-3
const ρᶜ = 1.0
const μᶜ = 1e-3
const uᶜ0 = 0.0  # Exact solution is valid only for uᶜ0 = 0.0

# Dispersed phase
const ⌀ᵈ = 1e-2
const ρᵈ = 1e3
const mᵈ = ρᵈ*π*⌀ᵈ^3/6
const τᵈ = ρᵈ*⌀ᵈ^2/(18*μᶜ*Cd)
const uᵈ0 = 1.0

# Composite types
abstract type AbstractExtrapolator end
struct ZeroExtrapolator <: AbstractExtrapolator end
struct ConstExtrapolator <: AbstractExtrapolator end
mutable struct LinearExtrapolator <: AbstractExtrapolator
    SUprevprev::Float64
end

mutable struct AsyncIntegrator{E <: AbstractExtrapolator}
    SUaccumTrue::Float64
    SUaccumExt::Float64
    SUprev::Float64
    timesWithoutSource::Int64
    maxTimesWithoutSource::Int64
    extrapolator::E
    vec::Vector{Float64}
end

function AsyncIntegrator(e::E) where E <: AbstractExtrapolator
    AsyncIntegrator{E}(
        0.0, 0.0, 0.0, 0, 0, e, [0.0]
    )
end

# Calculate analytical solution
function solve_exact(uᵈ0, uᶜ0, Δt, tEnd)
    f_uᵈ_exact(uᵈ0, t) = uᵈ0/(mᵈ + ρᶜ*Vᶜ)*(
        mᵈ + ρᶜ*Vᶜ*exp(-(mᵈ + ρᶜ*Vᶜ)/(ρᶜ*Vᶜ*τᵈ)*t)
    )
    f_uᶜ_exact(uᵈ0, t) = uᵈ0*mᵈ/(mᵈ + ρᶜ*Vᶜ)*(
        1.0 - exp(-(mᵈ + ρᶜ*Vᶜ)/(ρᶜ*Vᶜ*τᵈ)*t)
    )

    tRange = Δt:Δt:tEnd
    uSol = Vector{Vector{Float64}}(undef, length(tRange))
    for (i, t) in enumerate(tRange)
        sol = [f_uᵈ_exact(uᵈ0, t), f_uᶜ_exact(uᵈ0, t)]
        uSol[i] = sol
    end
    return uSol
end

function f_uᵈ(uᵈ, p, t)
    uᶜ, τᵈ = p
    return (uᶜ - uᵈ)/τᵈ
end

f_source(uᵈprev, uᵈ) = mᵈ*(uᵈprev - uᵈ)/(ρᶜ*Vᶜ)

# Extrapolate source with zero
function extrapolate!(integrator::AsyncIntegrator{ZeroExtrapolator})
    return zero(integrator.SUprev)
end

# Extrapolate source with the value from the previous time step
function extrapolate!(integrator::AsyncIntegrator{ConstExtrapolator})
    integrator.SUaccumExt += integrator.SUprev
    return integrator.SUprev
end

# General correction method
function correct!(integrator::AsyncIntegrator{<:AbstractExtrapolator}, SU)
    SUdelta = integrator.SUaccumTrue - integrator.SUaccumExt
    integrator.SUaccumTrue = zero(integrator.SUaccumTrue)
    integrator.SUaccumExt = zero(integrator.SUaccumExt)
    integrator.SUprev = SU
    return SUdelta
end

# Extrapolate source using linear extrapolation based on the two latest values
function extrapolate!(integrator::AsyncIntegrator{LinearExtrapolator})
    ext = integrator.extrapolator
    SUext = (2*integrator.SUprev - ext.SUprevprev)
    ext.SUprevprev = integrator.SUprev
    integrator.SUprev = SUext
    integrator.SUaccumExt += SUext
    return SUext
end

function correct!(integrator::AsyncIntegrator{LinearExtrapolator}, SU)
    integrator.extrapolator.SUprevprev = integrator.SUprev
    SUdelta = integrator.SUaccumTrue - integrator.SUaccumExt
    integrator.SUaccumTrue = zero(integrator.SUaccumTrue)
    integrator.SUaccumExt = zero(integrator.SUaccumExt)
    integrator.SUprev = SU
    return SUdelta
end

# Compute a time step of the Lagrangian phase using specified integrator
function evolve_carrier!(integrator::AsyncIntegrator, uᵈprev, uᵈ, uᶜ, t, tEnd, rng)
    SU = f_source(uᵈprev, uᵈ)
    integrator.SUaccumTrue += SU
    # isSourceAvailable = rand(rng, Bool)
    isSourceAvailable = rand(Bool)
    # isSourceAvailable = true
    if isSourceAvailable || t == tEnd
        # Corrector
        uᶜ += correct!(integrator, SU)
        integrator.maxTimesWithoutSource = max(
            integrator.maxTimesWithoutSource, integrator.timesWithoutSource
        )
        @show integrator.timesWithoutSource
        integrator.timesWithoutSource = 0
    else
        # Extrapolator
        uᶜ += extrapolate!(integrator)
        integrator.timesWithoutSource += 1
    end
    # @show integrator
    return uᶜ
end

# Solve the problem with async coupling
function solve_perturbed_coupled(uᵈ0, uᶜ0, Δt, tEnd, integrator)
    tRange = Δt:Δt:tEnd
    uSol = Vector{Vector{Float64}}(undef, length(tRange))
    rng = Xoshiro(1234)
    uᶜ = uᶜ0
    uᵈ = uᵈ0
    for (i, t) in enumerate(tRange)
        # Dispersed phase
        tspan = (0.0, Δt)
        pᵈ = ODEProblem{false}(f_uᵈ, uᵈ, tspan, [uᶜ, τᵈ], reltol = 1e-16, abstol = 1e-16)
        solᵈ = solve(pᵈ, Tsit5())
        uᵈprev = uᵈ
        uᵈ = solᵈ.u[end]

        # Carrier phase
        @show i
        uᶜ = evolve_carrier!(integrator, uᵈprev, uᵈ, uᶜ, t, tEnd, rng)
        uSol[i] = [uᵈ, uᶜ]
    end
    @show integrator.maxTimesWithoutSource
    return uSol
end

# Solve the problem with sync coupling, i.e. similar to the approach of OpenFOAM
function solve_seq_coupled(uᵈ0, uᶜ0, Δt, tEnd)
    uᶜ = uᶜ0
    uᵈ = uᵈ0
    tRange = Δt:Δt:tEnd
    uSol = Vector{Vector{Float64}}(undef, length(tRange))
    for (i, t) in enumerate(tRange)
        tspan = (0.0, Δt)
        pᵈ = ODEProblem(f_uᵈ, uᵈ, tspan, [uᶜ, τᵈ], reltol = 1e-16, abstol = 1e-16)
        solᵈ = solve(pᵈ, Tsit5())
        uᵈprev = uᵈ
        uᵈ = solᵈ.u[end]

        # Explicitly coupled with a Runge-Kutta method
        # f_uᶜ(uᶜ, p, t) = mᵈ*(uᵈ - uᶜ)/(ρᶜ*Vᶜ*τᵈ)
        # pᶜ = ODEProblem(f_uᶜ, uᶜ, tspan, reltol = 1e-16, abstol = 1e-16)
        # solᶜ = solve(pᶜ, Tsit5())

        # Explicit Euler
        uᶜ += f_source(uᵈprev, uᵈ)
        uSol[i] = [uᵈ, uᶜ]
    end
    return uSol
end

# Solve the problem using Julia's ODE solvers
function solve_unified(uᵈ0, uᶜ0, tEnd)
    function f_u(du, u, p, t)
        du[1] = (u[2] - u[1])/τᵈ  # duᵈ
        du[2] = mᵈ*(u[1] - u[2])/(ρᶜ*Vᶜ*τᵈ)  # duᶜ
    end
    u0 =[uᵈ0; uᶜ0]
    tspanUni = (0.0, tEnd)
    pUni = ODEProblem(f_u, u0, tspanUni)#, reltol = 1e-16, abstol = 1e-16)
    return solve(pUni, Tsit5())
end

totalMom0 = mᵈ*uᵈ0 + ρᶜ*Vᶜ*uᶜ0


solExact = solve_exact(uᵈ0, uᶜ0, Δt, tEnd)
totalMomExact = mᵈ*solExact[end][1] + ρᶜ*Vᶜ*solExact[end][2]
errorMomExact = totalMom0 - totalMomExact

solComp = solve_seq_coupled(uᵈ0, uᶜ0, Δt, tEnd)
totalMomComp = mᵈ*solComp[end][1] + ρᶜ*Vᶜ*solComp[end][2]
errorMomComp = totalMom0 - totalMomComp
errorSolComp = solExact - solComp

# integrator = AsyncIntegrator(ZeroExtrapolator())
integrator = AsyncIntegrator(ConstExtrapolator())
# integrator = AsyncIntegrator(LinearExtrapolator(0.0))
solPerturb = solve_perturbed_coupled(uᵈ0, uᶜ0, Δt, tEnd, integrator)
totalMomPerturb = mᵈ*solPerturb[end][1] + ρᶜ*Vᶜ*solPerturb[end][2]
errorMomPerturb = totalMom0 - totalMomPerturb
errorSolPerturb = solExact - solPerturb

solUni = solve_unified(uᵈ0, uᶜ0, tEnd)
totalMomUni = (mᵈ*solUni.u[end][1] + ρᶜ*Vᶜ*solUni.u[end][2])
errorUniExact = totalMom0 - totalMomUni
errorSolUni = solExact - solUni

# @show totalMom0 totalMomComp totalMomPerturb totalMomExact totalMomUni
# @show last(solExact) last(solComp) last(solPerturb) last(solUni)
@show last(errorSolComp)./last(solExact) last(errorSolPerturb)./last(solExact) last(errorSolUni)./last(solExact)
# @show errorMomExact errorMomComp errorUniExact
# @show errorMomExact/totalMom0 errorMomComp/totalMom0 errorMomPerturb/totalMom0 errorUniExact/totalMom0
@show last(errorSolPerturb) ./ last(errorSolComp)

# Plotting
using Plots
tRange = Δt:Δt:tEnd
plotlyjs()
# plot(tRange, hcat(solExact...)')
# plot!(tRange, hcat(solComp...)')
# plot!(tRange, hcat(solPerturb...)')

plot(tRange, hcat(errorSolComp...)')
plot!(tRange, hcat(errorSolPerturb...)')
