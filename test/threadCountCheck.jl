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
    Checks that the tracking does not depend on how many threads run it.

    Evolves a cloud a few steps through the synchronous driver on a frozen,
    seeded random velocity field and writes checksums of the parcel state and
    the momentum source.  A trajectory depends only on the frozen field, so
    the position and velocity sums have to be bitwise identical whatever the
    thread count.  The source sums need not be: parcels of one cell are
    accumulated in whatever order the threads reach them.

    Run twice with different thread counts and compare the two files against
    each other -- neither is a reference, so there is nothing here to keep in
    step with the code.

    Usage:
        julia --project=<root> threadCountCheck.jl checks_serial.txt
        julia --project=<root> --threads=4,1 threadCountCheck.jl checks_nt4.txt
        diff checks_serial.txt checks_nt4.txt
=#

# The library fixes the precision at include time.  Nothing here is shared
# with OpenFOAM, so the check is free to choose.
const scalar = Float64
const label = Int32

include(joinpath(@__DIR__, "../src/scaleTrack/scaleTrack.jl"))

println("Default-pool threads: ", Threads.nthreads(:default))

physics = StokesParticle(
    μᶜ = 1e-3,  # continuous phase dynamic viscosity
    ρᶜ = 1e3,   # continuous phase density
    ρᵈ = 1.0,   # disperse phase density
)

# The mesh is the check's own -- there is no case mesh to agree with
init_sync_tracking!(
    CPU(), physics;
    nParcels = 100_000,
    nCellsPerDirection = 40,
    origin = 0.0,
    ending = 1.0,
)

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
