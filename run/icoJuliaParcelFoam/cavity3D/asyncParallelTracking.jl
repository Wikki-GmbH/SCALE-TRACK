#=
    SCALE-TRACK

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# Case setup for the asynchronous particle tracking in the cavity3D case.
# All tracking code lives in the library; this script only
# sets the case parameters.

# The solver is built single precision (WM_PRECISION_OPTION=SP), which is the
# library default: scalar = Float32, label = Int32

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

executor = GPU()
# executor = CPU()

# Physical properties in SI units
physics = StokesFlow(
    μᶜ = 1e-3,  # continuous phase dynamic viscosity
    ρᶜ = 1e3,   # continuous phase density
    ρᵈ = 1.0,   # disperse phase density
    # These properties lead to large velocity source terms
    # μᶜ = 1e3,
    # ρᵈ = 1e8,
)

init_async_tracking!(
    executor, physics;
    # nParticles = 800_000_000  # RTX3090
    # nParticles = 25_000_000  # RTX3090 fast
    # nParticles = 1_000_000  # RTX3090 faster
    nParticles = 100_000,  # GT710
    nChunks = 10,

    # Mesh description; must be consistent with system/blockMeshDict
    nCellsPerDirection = 40,
    origin = 0.0,
    ending = 1.0,

    # Lagrangian decomposition by rank count.  The coefficients need to be
    # the same as in the decomposeParDict for the corresponding number of
    # ranks and must evenly divide the cell counts.
    decompositions = Dict(
        1 => (1, 1, 1),
        20 => (2, 2, 5)
    ),

    gcTimeStepInterval = 100,
)

# Test a few time steps when running in a REPL
if isinteractive()
    standalone_run!(2, 1e-3)
end
