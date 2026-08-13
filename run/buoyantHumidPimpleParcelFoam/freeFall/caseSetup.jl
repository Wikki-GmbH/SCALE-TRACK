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

# Case parameters of freeFall.  The library must be included before this file.
#
# One droplet released at rest near the top of a still 10 m column and falling
# to the floor: drag and gravity only.  Neither evaporation nor convective
# heat transfer is active, so the droplet keeps its mass, diameter and
# temperature and the trajectory is the whole result.  Allrun runs the
# OpenFOAM reference solver on the same case and greps the linear momentum of
# both for comparison.

# Physical properties in SI units
physics = HumidAirDroplet(
    # Evaporation and heat transfer are submodel choices here, so the
    # constants below keep their physical values and stay unused
    NoEvaporation(),
    NoHeatTransfer(),       # no convective exchange, as in the reference
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

nParcels = 1
nChunks = 1
nSubSteps = 10

# Mesh description; must be consistent with system/blockMeshDict
nCellsPerDirection = [1, 1, 10]
origin = [0.0, 0.0, 0.0]
ending = [1.0, 1.0, 10.0]

# The droplet starts at rest in the top cell, where the reference cloud
# injects
function init_droplets!(
    chunk,
    mesh,
    executor,
    randSeed = 19891,
    nChunksGlobal = 1
)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)

    fill!(c.boundingBox.min, 0.0)
    fill!(c.boundingBox.max, 0.0)
    fill!(c.X, 0.5SCL*mesh.L.x + mesh.origin.x)
    fill!(c.Y, 0.5SCL*mesh.L.y + mesh.origin.y)
    fill!(c.Z, 0.95SCL*mesh.L.z + mesh.origin.z)
    fill!(c.d, 500e-6SCL)
    fill!(c.props.T, 293.15SCL)
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    return nothing
end
