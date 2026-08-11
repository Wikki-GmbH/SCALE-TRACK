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

# Case setup for the asynchronous particle tracking in the hotRoom case:
# water droplets with heat and mass transfer (HumidAirDroplet physics) in a
# buoyant plume.  All tracking code lives in the library; this script and the
# case parameters beside it are the whole of the case setup.

# The solver is built single precision (WM_PRECISION_OPTION=SP), which is the
# library default: scalar = Float32, label = Int32

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

executor = GPU()
# executor = CPU()

include(joinpath(@__DIR__, "caseSetup.jl"))

init_async_tracking!(
    executor, physics;
    nParcelsTotal,
    nChunks,
    nCellsPerDirection, origin, ending,
    nSubSteps,

    # Lagrangian decomposition by rank count; must evenly divide the cell
    # counts.  Every rank count the case is run at needs an entry here.
    decompositions = Dict(
        1 => (1, 1, 1),
        2 => (2, 1, 1),
        4 => (1, 2, 2),
        8 => (2, 2, 2),
        32 => (4, 4, 2),
        64 => (4, 4, 4),
    ),

    # This is the case the cloud is scaled up in, so the timings are the
    # point of running it
    saveTimingsInterval = 10,

    gcTimeStepInterval = 100,
    extrapolator = ConstExtrapolator,
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
