#=
    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

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

#=
    Regression test for the HumidAirDroplet physics without OpenFOAM.

    Runs the synchronous driver on frozen, seeded random carrier fields and
    writes reproducible checksums of the particle state and the sources,
    which must match the golden file checks_humid.txt bitwise.

    The golden file is a regression guard on the physics as it stands: it
    says the numbers have not moved since it was written, not that they are
    right.  It began as a witness of the library port, having been produced
    by this procedure driven through the pre-library script of this case, and
    is regenerated whenever the physics is deliberately changed.

    Usage:
        julia --project=<env> testHumidTracking.jl checks_new.txt
        diff checks_humid.txt checks_new.txt
=#

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

executor = CPU()

include(joinpath(@__DIR__, "caseSetup.jl"))

init_sync_tracking!(
    executor, physics;
    nParcels,
    nCellsPerDirection, origin, ending,
    nSubSteps,
    initChunk! = init_droplets!,
)

# Frozen random carrier fields in physically sensible ranges
init_random!(reg["eulerian"].U, 2SCL, -1SCL)        # U ∈ [-1, 1] m/s
init_random!(reg["eulerian"].T, 10SCL, 290SCL)      # T ∈ [290, 300] K
init_random!(reg["eulerian"].rhoV, 0.01SCL, 0SCL)   # rhoV ∈ [0, 0.01] kg/m³

for _ in 1:5
    evolve_cloud(0.01SCL)
end

open(ARGS[1], "w") do io
    c = chunk
    for (name, arr) in (
        ("sumX", c.X), ("sumY", c.Y), ("sumZ", c.Z),
        ("sumU", c.U), ("sumV", c.V), ("sumW", c.W),
        ("sumT", c.props.T), ("sumD", c.d),
    )
        println(io, name, " ", sum(arr))
    end
    sUT = sum(reg["eulerian"].UTrans)
    println(io, "sumUTrans ", sUT.x, " ", sUT.y, " ", sUT.z)
    println(io, "sumHTrans ", sum(reg["eulerian"].hTrans))
    println(io, "sumRhoVTrans ", sum(reg["eulerian"].rhoVTrans))
end

println("Wrote ", ARGS[1])
