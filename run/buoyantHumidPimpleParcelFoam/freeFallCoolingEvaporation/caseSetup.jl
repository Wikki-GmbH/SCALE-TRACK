#=
    SCALE-TRACK

    Copyright (C) 2025-2026 Sergey Lesnik
    Copyright (C) 2025-2026 Henrik Rusche
    Copyright (C) 2025 Silvio Schmalfuß

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# Case parameters, shared by the coupled setup and the regression test.
# The library must be included before this file.

# Physical properties in SI units
physics = HumidAirDroplet(
    Evaporation();
    μᶜ = 1.8e-5,            # continuous phase dynamic viscosity
    ρᶜ = 1.2,               # continuous phase density
    ρᵈ = 1000.0,            # disperse phase density
    g = (0, 0, -9.81),      # gravitational acceleration
    Cₚᶜ = 1000,             # specific heat capacity of air
    Cₚᵈ = 4200,             # specific heat capacity of water
    Dᵈᶜ = 24.5e-6,          # diffusivity coefficient of water vapour in air
    Mᵈ = 18.01528e-3,       # molar mass of water
    σᶜ = 72.8e-3,           # surface tension of water in air
    RG = 8.3144598,         # gas constant
    SLH = 2.26471e6,        # specific latent heat of water vaporisation
)

nParcels = 10000
nChunks = 1
nSubSteps = 100

# Mesh description; must be consistent with system/blockMeshDict
nCellsPerDirection = [1, 1, 10]
origin = [0.0, 0.0, 0.0]
ending = [0.1, 0.1, 10.0]

# Droplets released in the top 10% of the column with uniform diameter and
# temperature
function init_droplets!(
    chunk,
    mesh,
    executor,
    randSeed = 19891,
    nChunksGlobal = 1
)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)

    rng = default_rng(executor)
    Random.seed!(rng, randSeed)
    fill!(c.boundingBox.min, 0.0)
    fill!(c.boundingBox.max, 0.0)
    rand!(rng, c.Z)
    fill!(c.X, 0.5SCL*mesh.L.x + mesh.origin.x)
    fill!(c.Y, 0.5SCL*mesh.L.y + mesh.origin.y)
    @. c.Z = 0.1SCL*c.Z*mesh.L.z + mesh.origin.z + 0.9SCL*mesh.L.z
    fill!(c.d, 500e-6SCL)
    fill!(c.props.T, 296.15SCL)
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    return nothing
end
