#=
    SCALE-TRACK

    Copyright (C) 2024-2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# Case setup for the synchronous single-rank tracking in the cavity3D test
# case.  All tracking code lives in the library; this script only sets the
# case parameters.

println("Load syncSerialTracking")

# This case is built against a double-precision OpenFOAM (WM_PRECISION_OPTION=DP)
const scalar = Float64
const label = Int32

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

# executor = GPU()
executor = CPU()  # this case does not require a GPU

# Physical properties in SI units
physics = StokesFlow(
    μᶜ = 1e-3,  # continuous phase dynamic viscosity
    ρᶜ = 1e3,   # continuous phase density
    ρᵈ = 1.0,   # disperse phase density
)

init_sync_tracking!(
    executor, physics;
    nParcels = 100_000,

    # Mesh description; must be consistent with the case's mesh
    nCellsPerDirection = 40,
    origin = 0.0,
    ending = 1.0,
)

# Create random velocity field when running in REPL
if isinteractive()
    randomize_velocity!()
end
