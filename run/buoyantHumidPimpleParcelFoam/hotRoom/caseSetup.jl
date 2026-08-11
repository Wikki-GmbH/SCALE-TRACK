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

# Case parameters of hotRoom, shared by the coupled setup and anything else
# driving the same cloud.
# The library must be included before this file.

# Physical properties in SI units
physics = HumidAirDroplet(
    Evaporation();
    μᶜ = 1.8e-5,            # continuous phase dynamic viscosity
    ρᶜ = 1.2,               # continuous phase density
    ρᵈ = 1000.0,            # disperse phase density
    # The droplets are a few micrometres across, so they settle at well under
    # a millimetre per second and simply follow the buoyant plume; the
    # carrier feels gravity through constant/g either way
    g = (0, 0, 0),          # gravitational acceleration
    Cₚᶜ = 1000,             # specific heat capacity of air
    Cₚᵈ = 4200,             # specific heat capacity of water
    Dᵈᶜ = 24.5e-6,          # diffusivity coefficient of water vapour in air
    Mᵈ = 18.01528e-3,       # molar mass of water
    σᶜ = 72.8e-3,           # surface tension of water in air
    RG = 8.3144598,         # gas constant
    SLH = 2.26471e6,        # specific latent heat of water vaporisation
)

# The size of the whole cloud, not of one chunk: this is the case the cloud
# is scaled up in, and it should stay the same size as the rank count varies.
# The reference cloud is sized to match, its injection covering the whole
# domain.
nParcelsTotal = 20_000
nChunks = 1
nSubSteps = 100

# Mesh description; must be consistent with system/blockMeshDict
nCellsPerDirection = [32, 32, 32]
origin = [0.0, 0.0, 0.0]
ending = [1.0, 1.0, 1.0]

# Droplets spread over the whole room, uniform in diameter over the range the
# reference cloud injects.  The positions follow the space-filling curve, so
# a chunk holds a connected region of the room and the carrier-field accesses
# of its kernel stay local.
function init_droplets!(chunk, mesh, executor, iChunk, nChunksGlobal)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)
    fill!(c.boundingBox.min, 0.0)
    fill!(c.boundingBox.max, 0.0)

    init_hilbert_positions!(chunk, mesh, executor, iChunk, nChunksGlobal)

    rng = default_rng(executor)
    rand!(rng, c.d)
    @. c.d = c.d*3e-6SCL + 2e-6SCL
    fill!(c.props.T, 288.15SCL)
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    return nothing
end
