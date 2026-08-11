#=
    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

#=
    Single-droplet verification of the HumidAirDroplet heat transfer.

    Drives the model's sub-step directly for one droplet in a frozen,
    uniform carrier: no mesh, no MPI, no GPU, no async coupling, no OpenFOAM.
    Whatever this reports is a property of the physics model alone.

    Two references are computed for every configuration:

      REF  the model's *own* ODE integrated in Float64 with RK4 at a step
           100x smaller than the production sub-step.  A gap between
           SCALE-TRACK and REF is integration error of the shipped scheme.

      EXP  the textbook Newton-cooling solution
           T(t) = Tc + (T0 - Tc)*exp(-t/tau),  tau = m*Cpd/(htc*As)
           with htc frozen at the initial state.  A gap between REF and EXP
           is the model's own nonlinearity (here: kappa(Td) and, in case B,
           the changing slip velocity), not an error.

    Case A  g = 0, carrier at rest, NoEvaporation.  Re = 0 makes Nu = 2
            exactly, so EXP is the exact solution up to the weak kappa(Td)
            drift.  This is the rigorous check.

    Case B  g = -9.81, NoEvaporation.  The droplet accelerates, Re and Nu
            grow, and EXP is only indicative -- REF is the reference.  This
            reproduces the conditions of the coupled free-fall case.

    Case C  as B but with Evaporation enabled, to show what the mass transfer
            adds on top.

    Usage:
        julia --project=<repo root> singleDropletCheck.jl [SP|DP]

    Default is DP so that the model error is not masked by Float32 roundoff;
    pass SP to reproduce the precision the solvers actually run in.
=#

const PRECISION = isempty(ARGS) ? "DP" : uppercase(ARGS[1])
const scalar = PRECISION == "SP" ? Float32 : Float64
const label = Int32

include(joinpath(@__DIR__, "../../src/scaleTrack/scaleTrack.jl"))

# ---------------------------------------------------------------- parameters
# The same droplet and carrier as the coupled free-fall case, repeated here so
# the check depends on the library alone and never on a case directory.
const D0   = 500e-6         # initial droplet diameter [m]
const TD0  = 296.15         # initial droplet temperature [K]
const TC   = 293.15         # carrier temperature [K]
const RHOV = 0.00865        # carrier vapour density [kg/m^3]
const MUC  = 1.8e-5
const RHOC = 1.2
const RHOD = 1000.0
const CPC  = 1000.0
const CPD  = 4200.0
const GZ   = -9.81

make_model(evap, gz) = HumidAirDroplet(
    evap;
    μᶜ = MUC, ρᶜ = RHOC, ρᵈ = RHOD, g = (0.0, 0.0, gz),
    Cₚᶜ = CPC, Cₚᵈ = CPD, Dᵈᶜ = 24.5e-6, Mᵈ = 18.01528e-3,
    σᶜ = 72.8e-3, RG = 8.3144598, SLH = 2.26471e6,
)

mass_of(d) = π*d^3*RHOD/6

# ------------------------------------------------- the model's own ODE (F64)
# The model's own heat-transfer and drag relations, written out so that REF
# solves the same equations, only accurately.
function coeffs(Td, w, d)
    κᶜ     = Td*8.9182e-5                     # at the droplet temperature
    Tsurf  = (2Td + TC)/3
    TRatio = TC/Tsurf
    ρˢ     = RHOC*TRatio
    μˢ     = MUC/TRatio
    κˢ     = κᶜ/TRatio
    Pr     = CPC*μˢ/κˢ
    urel   = -w                               # carrier at rest
    Re     = abs(urel)*d*ρˢ/μˢ
    Nu     = 2 + 0.6*sqrt(Re)*cbrt(Pr)
    htc    = Nu*κˢ/d
    As     = π*d^2
    m      = mass_of(d)
    bT     = htc*As/(m*CPD)                   # thermal 1/tau
    CdRe   = 24*(1 + Re^0.687*0.15)
    Fd     = m*0.75*MUC*CdRe/(RHOD*d^2)
    bU     = Fd/m
    return (bT = bT, bU = bU, Nu = Nu, Re = Re, htc = htc, Pr = Pr)
end

# state y = (Td, w); NoEvaporation so d and m are constant
function rhs(y, gz, d)
    Td, w = y
    c = coeffs(Td, w, d)
    return (c.bT*(TC - Td), c.bU*(0.0 - w) + gz*(1 - RHOC/RHOD))
end

function reference(gz, tEnd, dt)
    y = (TD0, 0.0)
    t = 0.0
    ts = [0.0]; Ts = [TD0]; ws = [0.0]
    nOut = max(1, round(Int, tEnd/dt/2000))
    for n in 1:round(Int, tEnd/dt)
        k1 = rhs(y, gz, D0)
        k2 = rhs((y[1] + dt/2*k1[1], y[2] + dt/2*k1[2]), gz, D0)
        k3 = rhs((y[1] + dt/2*k2[1], y[2] + dt/2*k2[2]), gz, D0)
        k4 = rhs((y[1] + dt*k3[1],   y[2] + dt*k3[2]),   gz, D0)
        y = (y[1] + dt/6*(k1[1] + 2k2[1] + 2k3[1] + k4[1]),
             y[2] + dt/6*(k1[2] + 2k2[2] + 2k3[2] + k4[2]))
        t += dt
        if n % nOut == 0
            push!(ts, t); push!(Ts, y[1]); push!(ws, y[2])
        end
    end
    return ts, Ts, ws
end

# ------------------------------------------------------- the shipped kernel
function run_scaletrack(model, Δt, nSteps)
    parcel = parcel_state(
        model, ScalarVec(0, 0, 0), ScalarVec(0, 0, 0), TD0, D0
    )
    carrier = (U = ScalarVec(0, 0, 0), T = scalar(TC), rhoV = scalar(RHOV))
    acc = (dUTrans = ScalarVec(0, 0, 0), dhTrans = 0SCL, drhoVTrans = 0SCL)
    ts = [0.0]; Ts = [TD0]; ws = [0.0]; ds = [D0]; hs = [0.0]
    nOut = max(1, nSteps ÷ 2000)
    for n in 1:nSteps
        parcel, acc = substep(model, parcel, carrier, acc, scalar(Δt))
        if n % nOut == 0
            push!(ts, n*Δt)
            push!(Ts, Float64(parcel.props.T))
            push!(ws, Float64(parcel.vel.z))
            push!(ds, Float64(parcel.props.⌀))
            push!(hs, Float64(acc.dhTrans))
        end
    end
    return ts, Ts, ws, ds, hs, acc, parcel
end

# ------------------------------------------------------------------ helpers
interp(ts, ys, t) = begin
    i = searchsortedfirst(ts, t)
    i <= 1 && return ys[1]
    i > length(ts) && return ys[end]
    f = (t - ts[i-1])/(ts[i] - ts[i-1])
    ys[i-1] + f*(ys[i] - ys[i-1])
end

# tau from an exponential fit through the endpoint: T(t) = Tc + dT0*exp(-t/tau)
tau_from(t, T) = begin
    r = (T - TC)/(TD0 - TC)
    (r <= 0 || r >= 1) ? NaN : -t/log(r)
end

banner(s) = (println(); println("="^74); println(s); println("="^74))

# ============================================================== case A: g = 0
function caseA(Δt, tEnd)
    banner("CASE A -- no gravity, carrier at rest, NoEvaporation  (Re=0, Nu=2)")
    model = make_model(NoEvaporation(), 0.0)
    nSteps = round(Int, tEnd/Δt)
    ts, Ts, ws, ds, hs, acc, parcel = run_scaletrack(model, Δt, nSteps)
    rts, rTs, rws = reference(0.0, tEnd, Δt/100)

    c0 = coeffs(TD0, 0.0, D0)
    τ0 = 1/c0.bT
    expT(t) = TC + (TD0 - TC)*exp(-t/τ0)

    println("  Nu (analytic, Re=0)      = ", round(c0.Nu, digits=6))
    println("  htc                      = ", round(c0.htc, digits=4), " W/m^2/K")
    println("  tau = m*Cpd/(htc*As)     = ", round(τ0, digits=5), " s")
    println()
    println("     t [s]    SCALE-TRACK        REF(RK4)      EXP(closed)   ",
            "ST-REF [K]   ST-EXP [K]")
    for t in (0.5, 1.0, 2.0, 5.0, 10.0)
        t > tEnd && continue
        st = interp(ts, Ts, t); rf = interp(rts, rTs, t); ex = expT(t)
        println("  ", lpad(t, 7), "   ",
                lpad(round(st, digits=9), 14), "  ", lpad(round(rf, digits=9), 14),
                "  ", lpad(round(ex, digits=9), 14),
                "  ", lpad(round(st-rf, sigdigits=3), 11),
                "  ", lpad(round(st-ex, sigdigits=3), 11))
    end
    errREF = maximum(abs(interp(ts, Ts, t) - interp(rts, rTs, t))
                     for t in range(0, tEnd, length=101))
    errEXP = maximum(abs(interp(ts, Ts, t) - expT(t))
                     for t in range(0, tEnd, length=101))
    println()
    println("  max |ST - REF| over the run = ", round(errREF, sigdigits=4), " K")
    println("  max |ST - EXP| over the run = ", round(errEXP, sigdigits=4), " K",
            "   (nonlinearity of kappa(Td), not an error)")

    # energy bookkeeping: the accumulated source must telescope to the change
    # in the droplet's absolute enthalpy Cpd*m*T
    m = mass_of(D0)
    expectedH = -CPD*(m*Float64(parcel.props.T) - m*TD0)
    println("  sum(dhTrans)                = ", Float64(acc.dhTrans))
    println("  -Cpd*(m*T_end - m*T_0)      = ", expectedH)
    println("  relative difference         = ",
            round(abs(Float64(acc.dhTrans) - expectedH)/abs(expectedH),
                  sigdigits=3))
    return errREF, τ0
end

# ================================================= case B/C: falling droplet
function caseBC(Δt, tEnd, evap, title)
    banner(title)
    model = make_model(evap, GZ)
    nSteps = round(Int, tEnd/Δt)
    ts, Ts, ws, ds, hs, acc, parcel = run_scaletrack(model, Δt, nSteps)
    rts, rTs, rws = reference(GZ, tEnd, Δt/100)       # NoEvaporation reference

    c0 = coeffs(TD0, 0.0, D0)
    println("     t [s]    SCALE-TRACK        REF(RK4)   ST-REF [K]     ",
            "w [m/s]      Re       Nu     tau_eff [s]")
    for t in (0.1, 0.25, 0.5, 1.0, 2.0)
        t > tEnd && continue
        st = interp(ts, Ts, t); rf = interp(rts, rTs, t)
        w  = interp(ts, ws, t)
        c  = coeffs(st, w, interp(ts, ds, t))
        println("  ", lpad(t, 7), "   ",
                lpad(round(st, digits=8), 14), "  ", lpad(round(rf, digits=8), 14),
                "  ", lpad(round(st-rf, sigdigits=3), 11),
                "  ", lpad(round(w, digits=4), 10),
                "  ", lpad(round(c.Re, digits=2), 7),
                "  ", lpad(round(c.Nu, digits=3), 7),
                "  ", lpad(round(1/c.bT, digits=4), 10))
    end
    errREF = maximum(abs(interp(ts, Ts, t) - interp(rts, rTs, t))
                     for t in range(0, tEnd, length=101))
    println()
    println("  max |ST - REF| over the run = ", round(errREF, sigdigits=4), " K")

    # How much has the droplet-to-gas temperature difference decayed?  This is
    # the quantity the coupled enthalpy source is proportional to.
    for t in (2.0,)
        t > tEnd && continue
        dT = interp(ts, Ts, t) - TC
        println("  (Td - Tc) at t=", t, ": ", round(dT, digits=5), " K  of ",
                TD0 - TC, " K  ->  decayed ",
                round((TD0 - TC)/dT, digits=3), "x")
        println("  implied tau                 = ",
                round(tau_from(t, interp(ts, Ts, t)), digits=4), " s")
    end
    if evap isa Evaporation
        println("  diameter  ", D0, " -> ", Float64(parcel.props.⌀), " m")
        println("  sum(drhoVTrans) [kg]        = ", Float64(acc.drhoVTrans))
    end
    return errREF
end

# ===================================================================== main
println("SCALE-TRACK single-droplet heat-transfer check")
println("precision: ", PRECISION, "  (scalar = ", scalar, ")")
println("droplet: d = ", D0, " m,  T0 = ", TD0, " K   carrier: Tc = ", TC, " K")

Δt = 1e-4      # the production sub-step

errA, τA = caseA(Δt, 10.0)
errB = caseBC(Δt, 2.0, NoEvaporation(),
              "CASE B -- gravity, NoEvaporation  (conditions of the coupled run)")
errC = caseBC(Δt, 2.0, Evaporation(),
              "CASE C -- gravity, Evaporation on  (REF is the no-evaporation ODE)")

banner("VERDICT")
# Two things bound how close the shipped scheme can come to REF, and the
# tolerance is whichever bound is larger.
#
# Truncation: the scheme is first-order semi-implicit Euler, so its error
# follows the ratio of the sub-step to the thermal relaxation time.  Expected,
# correct, and what the double-precision tolerance below is set from.
#
# Absorption: a sub-step moves the droplet temperature by (Dt/tau)*(Tc - Td),
# and once that falls below half the resolution of the stored temperature the
# addition returns the temperature unchanged and the droplet stops advancing.
# No arithmetic on an absolute temperature avoids this -- it is the storage,
# not the expression -- so the reachable gap is floored at ulp(T)/(2Dt/tau).
# In single precision that floor dominates; in double, truncation does.
#
# The floor grows with the number of sub-steps: a smaller sub-step is a
# smaller increment, absorbed at a larger gap.  Resolving the excursion finely
# and reaching it closely therefore pull against each other.
#
# The tolerance sits just above the floor, so the check still says the scheme
# reaches it and does no worse.
absorption = eps(scalar(TD0))/(2*Δt/τA)
tol = max(1e-3, 1.5*absorption)
println("  absorption floor ulp/(2Dt/tau) = ",
        round(absorption, sigdigits=3), " K")
println("  tolerance on |ST - REF|: ", round(tol, sigdigits=3), " K")
for (name, e) in (("A (g=0)", errA), ("B (falling)", errB))
    println("  case ", rpad(name, 12), " max|ST-REF| = ", rpad(round(e, sigdigits=4), 12),
            e < tol ? "PASS" : "FAIL")
end
println()
println("  Case C is reported for information only: its reference omits",
        " evaporation.")

# The exit status gates a run, so the test driver can read it; case C is
# informational and does not take part
exit((errA < tol && errB < tol) ? 0 : 1)
