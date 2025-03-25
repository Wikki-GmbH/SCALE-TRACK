nParticles = 50_000_000
nChunksPerDevice = 1
decomposition = (4, 4, 2)
reg["nTimeStepsWriteTimings"] = 10

# constants in SI units
# @SiSc: added additional properties
μᶜ = 1.8e-5SCL              # continuous phase dynamic viscosity
ρᶜ = 1.2SCL                 # continuous phase density
ρᵈ = 1000.0SCL              # disperse phase density
g = ScalarVec(0, 0, 0)      # gravitational acceleration
Cₚᶜ = 1000SCL               # specific heat capacity of air
Cₚᵈ = 4200SCL               # specific heat capacity of water
Dᵈᶜ = 24.5e-6SCL            # diffusivity coefficient of water vapour in air
Mᵈ = 18.01528e-3SCL         # molar mass of water
σᶜ = 72.8e-3SCL             # surface tension of water in air
RG = 8.3144598SCL           # gas constant
SLH = 2264.71SCL            # specific latent heat of water vaporisation

nCellsPerDirection = [32, 32, 32]
nCells = prod(nCellsPerDirection)
origin = [0.0, 0.0, 0.0]
ending = [1.0, 1.0, 1.0]
