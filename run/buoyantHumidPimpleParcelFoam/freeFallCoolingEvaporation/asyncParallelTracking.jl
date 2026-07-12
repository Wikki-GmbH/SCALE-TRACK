#=
    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik
    Copyright (C) 2026 Henrik Rusche

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# Case setup for the asynchronous particle tracking in the
# freeFallCoolingEvaporation case: water droplets with heat and mass transfer
# (HumidAirDroplet physics).  All tracking code lives in the library; this
# script and the case parameters beside it are the whole of the case setup.

# The solver is built single precision (WM_PRECISION_OPTION=SP), which is the
# library default: scalar = Float32, label = Int32

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

executor = GPU()
# executor = CPU()

include(joinpath(@__DIR__, "caseSetup.jl"))

init_async_tracking!(
    executor, physics;
    nParticles,
    nChunks,
    nCellsPerDirection, origin, ending,
    nSubSteps,

    # Lagrangian decomposition by rank count; must evenly divide the cell
    # counts.  (The pre-library script used (2, 1, 1) for any parallel run,
    # which is inconsistent with the 1x1x10 mesh.)
    decompositions = Dict(
        1 => (1, 1, 1),
        2 => (1, 1, 2)
    ),

    gcTimeStepInterval = 20,
    initChunk! = init_droplets!,
)

# Test a few time steps when running in a REPL.  The velocity field is
# randomized; temperature and vapour density must be set to sensible ranges.
if isinteractive()
    for i in eachindex(reg["eulerian"])
        # Slaves allocate only their own partition
        isassigned(reg["eulerian"], i) || continue
        init_random!(reg["eulerian"][i].T, 10SCL, 290SCL)
        init_random!(reg["eulerian"][i].rhoV, 0.01SCL, 0SCL)
    end
    standalone_run!(2, 1e-3)
end
