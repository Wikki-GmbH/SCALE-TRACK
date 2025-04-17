# General
nCellsPerDirection = 40
decomposition = (2, 2, 5)
nParticles = 50_000_000
nChunksPerDevice = 100

# Geometry
origin = 0.0
ending = 1.0

# Physics
μᶜ = 1e-3
ρᶜ = 1e3
ρᵈ = 1.0
# These properties lead to large velocity source terms
# μᶜ = 1e3
# ρᵈ = 1e8

# Derivate
nCells = nCellsPerDirection^3

