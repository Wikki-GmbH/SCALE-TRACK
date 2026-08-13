#=
    SPDX-License-Identifier: GPL-3.0-or-later

    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik
    Copyright (C) 2026 Henrik Rusche

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# library default: scalar = Float32, label = Int32

# Report the parcel state every coupling step, as the reference cloud does
const CloudLogFrequency = 1

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

executor = GPU()
# executor = CPU()

include(joinpath(@__DIR__, "caseSetup.jl"))

init_async_tracking!(
    executor, physics;
    nParcels,
    nChunks,
    nCellsPerDirection, origin, ending,
    nSubSteps,

    # A single cell admits only the trivial decomposition
    decompositions = Dict(1 => (1, 1, 1)),

    # Hold the previous true source between coupling steps
    extrapolator = ConstExtrapolator,

    gcTimeStepInterval = 100,
    initChunk! = init_droplets!,
)

# Test a few time steps when running in a REPL
if isinteractive()
    for i in eachindex(reg["eulerian"])
        # Slaves allocate only their own partition
        isassigned(reg["eulerian"], i) || continue
        init_random!(reg["eulerian"][i].T, 0SCL, 293.15SCL)
        init_random!(reg["eulerian"][i].rhoV, 0.01SCL, 0SCL)
    end
    standalone_run!(2, 1e-3)
end
