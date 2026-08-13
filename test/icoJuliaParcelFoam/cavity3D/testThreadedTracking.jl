#=
    SPDX-License-Identifier: GPL-3.0-or-later

    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

#=
    Regression test for the CPU particle tracking without OpenFOAM.

    Runs a few evolve steps on a frozen, seeded random velocity field and
    writes reproducible checksums of the particle state and the momentum
    source.  Particle trajectories depend only on the (frozen) velocity
    field, so the position/velocity sums must be bitwise identical for any
    thread count; the momentum-source sums may differ by floating-point
    roundoff only (different per-cell accumulation order).

    Usage:
        julia --project=. testThreadedTracking.jl checks_serial.txt
        julia --project=. --threads=4,1 testThreadedTracking.jl checks_nt4.txt
        diff checks_serial.txt checks_nt4.txt
=#

include("./syncSerialTracking.jl")

println("Default-pool threads: ", Threads.nthreads(:default))

init_random!(reg["eulerian"].U, 2SCL, -1SCL)

for _ in 1:5
    evolve_cloud(0.01)
end

open(ARGS[1], "w") do io
    c = chunk
    for (name, arr) in (
        ("sumX", c.X), ("sumY", c.Y), ("sumZ", c.Z),
        ("sumU", c.U), ("sumV", c.V), ("sumW", c.W),
    )
        println(io, name, " ", sum(arr))
    end
    sUT = sum(reg["eulerian"].UTrans)
    println(io, "sumUTrans ", sUT.x, " ", sUT.y, " ", sUT.z)
end

println("Wrote ", ARGS[1])
