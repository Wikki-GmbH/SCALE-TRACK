#=
    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# What Allcheck compares for singleCellEvaporation, and how closely.
#
# One droplet evaporating into still air that starts at its own temperature.
# It cools to its wet bulb, and the parcel temperature is judged over that
# excursion rather than in absolute terms: the two sides resolve the cooling
# with different numbers of Lagrangian sub-steps.

include(joinpath(@__DIR__, "../../../test/caseCompare.jl"))

quantities = [
    Quantity("parcel temperature", "Tp_OF.csv", "Tp_ST.csv", :abs, 0.5, "K"),
    Quantity("parcel mass, relative",
        "H2OPMass_OF.csv", "H2OPMass_ST.csv", :rel, 1e-3),
    Quantity("vapour mass, relative",
        "H2OVMass_OF.csv", "H2OVMass_ST.csv", :rel, 5e-4),
]

banner("singleCellEvaporation")
require(files_of(quantities)...)

for q in quantities
    check!(q)
end

# The reference cloud reports no diameter, so this one is shown, not judged
if isfile("dp_ST.csv") && filesize("dp_ST.csv") > 0
    d = read_series("dp_ST.csv")
    println()
    println("  droplet diameter, tracking only (the reference reports none)")
    println("    ", first(d)[1], " -> ", last(d)[1], " m")
end

finish()
