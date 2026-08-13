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
    Checks the GPU executor against the CPU executor.

    Both executors are driven through the synchronous driver over the same
    frozen random carrier field, starting from the *same* particle state:
    the chunk is initialized on the host and copied to the device, because
    the two executors draw from different random number generators and a
    per-executor init would not be comparable.

    The particle state is pure advection and must agree to roundoff.  The
    momentum source is accumulated with atomics, so its per-cell summation
    order differs between the executors and between runs; it is compared
    with a looser tolerance.

    Needs a GPU.  Synchronous mode only: no MPI, no async, no OpenFOAM.

    Both physics models are covered: StokesParticle exercises the momentum
    source alone, HumidAirDroplet additionally the thermal energy and mass
    sources, i.e. the scalar-field atomics next to the vector-field ones.

    Usage:
        julia --project=<repo root> gpuExecutorCheck.jl [SP|DP] [CUDA|ROCm] \
                                                       [stokes|humid|both]
=#

const PRECISION = isempty(ARGS) ? "SP" : uppercase(ARGS[1])
const scalar = PRECISION == "SP" ? Float32 : Float64
const label = Int32
const gpuBackend =
    (length(ARGS) >= 2 && lowercase(ARGS[2]) == "rocm") ? :ROCm : :CUDA
const MODELARG = length(ARGS) >= 3 ? lowercase(ARGS[3]) : "both"

include(joinpath(@__DIR__, "../../src/scaleTrack/scaleTrack.jl"))

# ---------------------------------------------------------------- parameters
const NPARCELS = 100_000
const NCELLS   = 40          # cells per direction
const NSTEPS   = 5
const DT       = 0.01

# Advection is deterministic, so the executors may differ by roundoff only.
#
# The per-cell source is a much blunter instrument than it looks.  A parcel
# sitting within roundoff of a cell face can be located in either of the two
# cells, and its whole contribution then moves with it; with a parcel count
# of the order of the cell count that displaces a large fraction of a cell's
# source.  The per-cell tolerance is therefore a loose regression guard, and
# loosest in single precision.
#
# What reassignment cannot perturb is the total over all cells, so the sum of
# each source field is compared separately and tightly -- that is the
# conservation statement the coupling actually rests on.
const TOL_STATE      = PRECISION == "SP" ? 1e-5 : 1e-12
const TOL_SOURCE     = PRECISION == "SP" ? 1e-1 : 1e-8
const TOL_SOURCE_SUM = PRECISION == "SP" ? 1e-4 : 1e-11

# Carrier state of the humid case, frozen
const TC   = 293.15      # carrier temperature [K]
const RHOV = 0.00865     # carrier vapour density [kg/m^3]
const TD0  = 296.15      # initial droplet temperature [K]

make_model(::Val{:stokes}) = StokesParticle(
    μᶜ = 1e-3,  # continuous phase dynamic viscosity
    ρᶜ = 1e3,   # continuous phase density
    ρᵈ = 1.0,   # disperse phase density
)

make_model(::Val{:humid}) = HumidAirDroplet(
    Evaporation();
    μᶜ = 1.8e-5, ρᶜ = 1.2, ρᵈ = 1000.0, g = (0.0, 0.0, -9.81),
    Cₚᶜ = 1000.0, Cₚᵈ = 4200.0, Dᵈᶜ = 24.5e-6, Mᵈ = 18.01528e-3,
    σᶜ = 72.8e-3, RG = 8.3144598, SLH = 2264.71,
)

# The host chunk both executors start from, built once per model
hostRef = nothing

function init_from_host!(chunk, mesh, executor, randSeed = 19891)
    global hostRef, physics
    if isnothing(hostRef)
        h = allocate_chunk(CPU(), physics, chunk.N, chunk.nSubSteps)
        init!(h, mesh, CPU(), randSeed)
        # The droplet temperature would otherwise start at zero
        haskey(h.props, :T) && fill!(h.props.T, TD0*SCL)
        hostRef = h
    end
    copy!(chunk, hostRef)
    return nothing
end

function run_case(executorArg)
    init_sync_tracking!(
        executorArg, physics;
        nParcels = NPARCELS,
        nCellsPerDirection = NCELLS,
        origin = 0.0,
        ending = 1.0,
        initChunk! = init_from_host!,
    )

    # Frozen carrier: seeded on the host, identical for both executors
    e = reg["eulerian"]
    init_random!(e.U, 2SCL, -1SCL)
    if hasproperty(e, :T)
        fill!(e.T, TC*SCL)
        fill!(e.rhoV, RHOV*SCL)
    end

    for _ in 1:NSTEPS
        evolve_cloud(DT)
    end

    c = chunk
    state = Dict{String, Vector{Float64}}(
        "X" => Array(c.X), "Y" => Array(c.Y), "Z" => Array(c.Z),
        "U" => Array(c.U), "V" => Array(c.V), "W" => Array(c.W),
        "d" => Array(c.d),
    )
    for (name, arr) in pairs(c.props)
        state[string(name)] = Array(arr)
    end

    sources = Dict{String, Vector{Float64}}()
    for f in source_fields(typeof(e))
        arr = getfield(e, f)
        sources[string(f)] =
            eltype(arr) <: ScalarVec ?
            reduce(vcat, [[v.x, v.y, v.z] for v in arr]) : Vector{Float64}(arr)
    end
    return state, sources
end

# Relative difference, normalized by the magnitude of the reference field so
# that near-zero entries do not dominate
function max_rel_diff(a, b)
    scale = maximum(abs, a)
    scale = scale == 0 ? one(eltype(a)) : scale
    return maximum(abs.(a .- b)) / scale
end

function check_model(modelName)
    global physics, hostRef
    physics = make_model(Val(modelName))
    hostRef = nothing

    println("\n", "="^78)
    println("GPU executor check -- $(modelName), $(PRECISION), $(gpuBackend)")
    println("="^78)

    println("\n--- CPU reference ---")
    cpuState, cpuSources = run_case(CPU())

    println("\n--- GPU ---")
    gpuState, gpuSources = run_case(GPU())

    println("\n", "-"^78)
    println("RESULTS -- $(modelName)")
    println("-"^78)

    # Collected before reducing, so every field is reported even once one
    # of them has failed
    stateOk =
        map(sort(collect(keys(cpuState)))) do name
            d = max_rel_diff(cpuState[name], gpuState[name])
            ok = d <= TOL_STATE
            println("  parcel ", rpad(name, 14), "max rel diff = ",
                rpad(round(d; sigdigits = 4), 12), ok ? "PASS" : "FAIL")
            return ok
        end |> all

    sourceOk =
        map(sort(collect(keys(cpuSources)))) do name
            a, b = cpuSources[name], gpuSources[name]
            d = max_rel_diff(a, b)
            sa, sb = sum(a), sum(b)
            ds = abs(sa - sb) / max(abs(sa), eps())
            ok = d <= TOL_SOURCE && ds <= TOL_SOURCE_SUM
            println("  source ", rpad(name, 14),
                "per cell = ", rpad(round(d; sigdigits = 4), 12),
                "sum = ", rpad(round(ds; sigdigits = 4), 12),
                ok ? "PASS" : "FAIL")
            return ok
        end |> all

    println("\n  tolerances: state ", TOL_STATE, ", source per cell ",
        TOL_SOURCE, ", source sum ", TOL_SOURCE_SUM)
    ok = stateOk && sourceOk
    println(
        ok ?
        "  PASS -- the GPU executor reproduces the CPU executor." :
        "  FAIL -- the executors disagree beyond roundoff."
    )
    return ok
end

models = MODELARG == "both" ? (:stokes, :humid) : (Symbol(MODELARG),)
allOk = all(check_model(m) for m in models)

println("\n", "="^78)
println(allOk ? "ALL PASS" : "FAILURES -- see above")
println("="^78)
allOk || exit(1)
