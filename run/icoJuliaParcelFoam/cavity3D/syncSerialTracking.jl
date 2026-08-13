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

# Case setup for the synchronous single-rank tracking in the cavity3D case.
# All tracking code lives in the library; this script only
# sets the case parameters.

println("Load syncSerialTracking")

# The solver is built single precision (WM_PRECISION_OPTION=SP), which is the
# library default: scalar = Float32, label = Int32

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

executor = GPU()
# executor = CPU()

# Physical properties in SI units
physics = StokesParticle(
    μᶜ = 1e-3,  # continuous phase dynamic viscosity
    ρᶜ = 1e3,   # continuous phase density
    ρᵈ = 1.0,   # disperse phase density
)

init_sync_tracking!(
    executor, physics;
    nParcels = 100_000,

    # Mesh description; must be consistent with the case's mesh
    nCellsPerDirection = 20,
    origin = 0.0,
    ending = 1.0,
)

# Create random velocity field when running in REPL
if isinteractive()
    randomize_velocity!()
end
