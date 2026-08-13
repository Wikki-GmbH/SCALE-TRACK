#=
    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

# What Allcheck compares for freeFallCoolingEvaporation, and how closely.
#
# The only case running evaporation and heat transfer together, so it is the
# one that measures the thermal energy source.  Everything compared here is a
# carrier quantity: the two solvers report no common parcel quantity.
#
# The carrier momentum passes through zero as the droplets decelerate the air
# they fall through, so it is judged against its own peak rather than row by
# row.

include(joinpath(@__DIR__, "../../../test/caseCompare.jl"))

quantities = [
    Quantity("carrier temperature",
        "minmaxT_OF.csv", "minmaxT_ST.csv", :abs, 0.05, "K"),
    Quantity("carrier vapour density",
        "minmaxrhoV_OF.csv", "minmaxrhoV_ST.csv", :abs, 5e-5, "kg/m^3"),
    Quantity("carrier momentum, of peak",
        "integralMomentum_OF.csv", "integralMomentum_ST.csv",
        :peak, 0.03),
    Quantity("vapour mass, relative",
        "integralrhoV_OF.csv", "integralrhoV_ST.csv", :rel, 5e-4),
]

banner("freeFallCoolingEvaporation")
require(files_of(quantities)...)

for q in quantities
    check!(q)
end

finish()
