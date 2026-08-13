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
    Checks the tracking driver against the physics kernel.

    Drives nParcels *identical* droplets through the synchronous driver in
    a frozen, uniform carrier and compares the per-step sources against the
    same kernel called directly and multiplied by nParcels.  Since every
    droplet is identical and the carrier is uniform, the two must agree to
    roundoff; any difference is the driver -- mesh location, cell crossing,
    source accumulation and flushing, or the per-step reset.

    Synchronous mode only: no MPI, no async, no GPU, no OpenFOAM.  The
    carrier is never updated from the sources, so it stays frozen.

    Both models run in one process, which also covers the driver rebuilding
    its cached task buffers on re-initialisation.

    Usage:
        julia --project=<repo root> syncDriverCheck.jl [SP|DP] [evap|noevap]

    Without the second argument both models are run.
=#

const PRECISION = isempty(ARGS) ? "DP" : uppercase(ARGS[1])
const EVAPARG = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "both"
const scalar = PRECISION == "SP" ? Float32 : Float64
const label = Int32

include(joinpath(@__DIR__, "../../src/scaleTrack/scaleTrack.jl"))

# ---------------------------------------------------------------- parameters
const TC   = 293.15        # carrier temperature [K]        (frozen)
const RHOV = 0.00865       # carrier vapour density [kg/m^3](frozen)
const TD0  = 296.15        # initial droplet temperature [K]
const D0   = 500e-6        # initial droplet diameter [m]
const RHOD = 1000.0
const CPD  = 4200.0
const Z0   = 9.5           # mid of the top cell of the 1x1x10 mesh

const NPARCELS = 10_000     # as in the coupled free-fall case
const NSUB = 100        # as in the coupled free-fall case
const NCELLS = [1, 1, 10]
const ORIGIN = [0.0, 0.0, 0.0]
const ENDING = [0.1, 0.1, 10.0]
const DT = 0.01       # coupling time step
const NSTEPS = 200        # two seconds of physical time

make_model(evap) = HumidAirDroplet(
    evap;
    μᶜ = 1.8e-5, ρᶜ = 1.2, ρᵈ = RHOD, g = (0.0, 0.0, -9.81),
    Cₚᶜ = 1000.0, Cₚᵈ = CPD, Dᵈᶜ = 24.5e-6, Mᵈ = 18.01528e-3,
    σᶜ = 72.8e-3, RG = 8.3144598, SLH = 2.26471e6,
)

# All droplets identical and co-located, so the ensemble is exactly
# nParcels copies of one droplet
function init_identical!(chunk, mesh, executor)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)
    fill!(c.boundingBox.min, 0.0)
    fill!(c.boundingBox.max, 0.0)
    fill!(c.X, 0.5SCL*mesh.L.x + mesh.origin.x)
    fill!(c.Y, 0.5SCL*mesh.L.y + mesh.origin.y)
    fill!(c.Z, Z0*SCL)
    fill!(c.d, D0*SCL)
    fill!(c.props.T, TD0*SCL)
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    return nothing
end

# ------------------------------------------------- reference: kernel, direct
# One droplet, same frozen carrier, nSubSteps of Dt/nSubSteps per Euler step.
# Returns per-Euler-step values so the driver can be compared step by step
# rather than only at the end.
function kernel_reference(model)
    parcel = parcel_state(
        model, ScalarVec(0.5*0.1, 0.5*0.1, Z0), ScalarVec(0, 0, 0), TD0, D0
    )
    carrier = (U = ScalarVec(0, 0, 0), T = scalar(TC), rhoV = scalar(RHOV))
    Δt = scalar(DT/NSUB)
    Td = Float64[]
    src = Float64[]
    zs = Float64[]
    msrc = Float64[]
    ds = Float64[]
    for n in 1:NSTEPS
        acc = (dUTrans = ScalarVec(0, 0, 0), dhTrans = 0SCL, drhoVTrans = 0SCL)
        for s in 1:NSUB
            parcel, acc = substep(model, parcel, carrier, acc, Δt)
        end
        push!(Td, Float64(parcel.props.T))
        push!(src, NPARCELS*Float64(acc.dhTrans))
        push!(msrc, NPARCELS*Float64(acc.drhoVTrans))
        push!(zs, Float64(parcel.pos.z))
        push!(ds, Float64(parcel.props.⌀))
    end
    return Td, src, zs, msrc, ds
end

# ------------------------------------------------------- the library driver
function driver_run(model)
    init_sync_tracking!(
        CPU(), model;
        nParcels = NPARCELS,
        nCellsPerDirection = NCELLS, origin = ORIGIN, ending = ENDING,
        nSubSteps = NSUB,
        initChunk! = init_identical!,
    )

    # Frozen uniform carrier.  Never updated from the sources, so it stays
    # exactly what the kernel reference sees.
    e = reg["eulerian"]
    fill!(e.U, ScalarVec(0, 0, 0))
    fill!(e.T, scalar(TC))
    fill!(e.rhoV, scalar(RHOV))
    fill!(e.UTrans, ScalarVec(0, 0, 0))
    fill!(e.hTrans, 0SCL)
    fill!(e.rhoVTrans, 0SCL)

    Td = Float64[]
    src = Float64[]
    zs = Float64[]
    msrc = Float64[]
    ds = Float64[]
    for n in 1:NSTEPS
        evolve_cloud(scalar(DT))
        push!(Td, Float64(chunk.props.T[1]))
        push!(src, Float64(sum(reg["eulerian"].hTrans)))
        push!(msrc, Float64(sum(reg["eulerian"].rhoVTrans)))
        push!(zs, Float64(chunk.Z[1]))
        push!(ds, Float64(chunk.d[1]))
    end
    return Td, src, zs, msrc, ds
end

# ===================================================================== main
function compare(evap, title)
    println()
    println("="^78)
    println(title)
    println("="^78)

    kTd, kSrc, kZ, kM, kD = kernel_reference(make_model(evap))
    dTd, dSrc, dZ, dM, dD = driver_run(make_model(evap))

    println()
    println("  step   t[s]   Td kernel     Td driver      dTd [K]    ",
        "src kernel   src driver     ratio")
    for n in (1, 20, 50, 100, 180, 200)
        n > NSTEPS && continue
        println("  ", lpad(n, 4), "  ", lpad(round(n*DT, digits = 2), 5), "  ",
            lpad(round(kTd[n], digits = 6), 11), "  ",
            lpad(round(dTd[n], digits = 6), 11), "  ",
            lpad(round(dTd[n]-kTd[n], sigdigits = 3), 11), "  ",
            lpad(round(kSrc[n], sigdigits = 5), 11), "  ",
            lpad(round(dSrc[n], sigdigits = 5), 11), "  ",
            lpad(round(dSrc[n]/kSrc[n], digits = 5), 9))
    end

    maxdT  = maximum(abs.(dTd .- kTd))
    maxrel = maximum(abs.(dSrc .- kSrc) ./ abs.(kSrc))
    maxdZ  = maximum(abs.(dZ .- kZ))
    println()
    println(
        "  max |Td_driver - Td_kernel|     = ",
        round(maxdT, sigdigits = 4),
        " K"
    )
    println(
        "  max |z_driver  - z_kernel|      = ",
        round(maxdZ, sigdigits = 4),
        " m"
    )
    println(
        "  max relative source difference  = ",
        round(maxrel, sigdigits = 4)
    )
    println("  decay of (Td - Tc) over the run : kernel ",
        round((TD0-TC)/(kTd[end]-TC), digits = 3), "x   driver ",
        round((TD0-TC)/(dTd[end]-TC), digits = 3), "x")

    # The quantities the coupled solver logs, for direct comparison with a
    # coupled run
    println()
    println("  --- as the coupled solver logs them (per Euler step) ---")
    println("   t[s]   integral hTrans [J]   integral rhoVTrans [kg]   d [m]")
    for n in (20, 100, 180, 200)
        n > NSTEPS && continue
        println("  ", lpad(round(n*DT, digits = 2), 5), "  ",
            lpad(round(dSrc[n], sigdigits = 5), 18), "  ",
            lpad(round(dM[n], sigdigits = 5), 22), "  ",
            lpad(round(dD[n], sigdigits = 6), 12))
    end
    return maxdT, maxrel
end

println("SCALE-TRACK sync-driver vs physics-kernel check")
println("precision: ", PRECISION, "  (scalar = ", scalar, ")")
println(NPARCELS, " identical droplets, frozen uniform carrier at ", TC, " K")

# Both models in one process by default: that re-initializes the driver and so
# also exercises the invalidation of its cached Eulerian copy
models =
    EVAPARG == "noevap" ? [("noevap", NoEvaporation())] :
    EVAPARG == "evap"   ? [("evap", Evaporation())]     :
    [("noevap", NoEvaporation()), ("evap", Evaporation())]

results = [
    (
        name,
        compare(
            m,
            name == "noevap" ? "NoEvaporation" :
            "Evaporation  (the setting of the coupled case)"
        )
    )
    for (name, m) in models
]

println();
println("="^78);
println("VERDICT");
println("="^78)
tol = PRECISION == "SP" ? 1e-3 : 1e-9
println("  tolerance on the relative source difference: ", tol)
for (name, r) in results
    println("  ", rpad(name, 16), "max rel src diff = ",
        rpad(round(r[2], sigdigits = 4), 12), r[2] < tol ? "PASS" : "FAIL")
end
println()
println("  PASS here means the driver faithfully reproduces the kernel, so a")
println("  discrepancy seen in the coupled solver comes from the async path,")
println("  the carrier handoff or the OpenFOAM side -- not from this layer.")

# The exit status gates a run, so the test driver can read it
exit(all(r[2] < tol for (_, r) in results) ? 0 : 1)
