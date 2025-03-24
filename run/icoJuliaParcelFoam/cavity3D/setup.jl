nParticles = 50_000_000
nChunksPerDevice = 100

μᶜ = 1e-3
ρᶜ = 1e3
ρᵈ = 1.0
# These properties lead to large velocity source terms
# μᶜ = 1e3
# ρᵈ = 1e8
nCellsPerDirection = 40
nCells = nCellsPerDirection^3
origin = 0.0
ending = 1.0

decomposition = (2, 2, 5)
