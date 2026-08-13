#=
    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# What Allcheck compares for freeFall, and how closely.
#
# One droplet falling to the floor under drag and gravity, with neither
# evaporation nor heat transfer.  Two things need more than a row-by-row
# comparison, so this case does its own rather than listing quantities.

include(joinpath(@__DIR__, "../../../test/caseCompare.jl"))

banner("freeFall")
require("momentumZ_OF.csv", "momentumZ_ST.csv",
    "H2OPMass_OF.csv", "H2OPMass_ST.csv")

# The reference reports the momentum as a vector, of which the third
# component is the falling direction; the tracking reports that component
# alone
of = read_series("momentumZ_OF.csv"; col = 3)
st = read_series("momentumZ_ST.csv")

# The first row at which a series stops falling.  Only the phase before it is
# comparable: the momentum there is a function of the time since release,
# which both solvers start from rest, while past the bounce the two are out of
# phase and describe different states.
function bounce(s)
    for i in 2:length(s)
        s[i][1] >= 0 && return i
    end
    return length(s) + 1
end

bOF, bST = bounce(of), bounce(st)
falling = min(bOF, bST) - 1

println()
println("  floor bounce: OF at row ", bOF, ", ST at row ", bST)
println()
println("  linear momentum z over the falling phase, rows 1..", falling)
best, shift = scan(
    of, st, :peak; unit = "of peak", last = falling, lastB = bST - 1
)
verdict!("momentum z, of peak", best, 0.01)

# Both sides release the droplet at the same point, so they fall the same
# distance and what is left once the coupling offset is taken out is
# accumulated integration error
verdict!("bounce row beyond the offset", abs(bST - shift - bOF), 4)

# With the evaporation off the mass is a constant of the run on both sides,
# so no shift applies: the drift of each series and their difference are the
# whole statement
mOF = read_series("H2OPMass_OF.csv")
mST = read_series("H2OPMass_ST.csv")
drift(s) =
    let v = [r[1] for r in s]
        (maximum(v) - minimum(v))/maximum(v)
    end

println()
println("  mass in the system, ", first(mOF)[1], " kg")
verdict!("mass drift OF", drift(mOF), 1e-5)
verdict!("mass drift ST", drift(mST), 1e-5)
verdict!("mass OF vs ST", difference(mOF, mST, 0, :rel)[2], 1e-5)

finish()
