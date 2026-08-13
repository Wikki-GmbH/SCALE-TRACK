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

# What Allcheck compares for hotRoom, and how closely.
#
# This is the case the cloud is scaled up in, and what it exercises is the
# machinery at size: chunks over several tracking masters, the cloud laid out
# along the space-filling curve, and the step timings.  The physics it adds
# over the other cases is little -- droplets of a few micrometres carry so
# little water that the room barely notices them -- so what is compared here
# is that the coupling leaves the carrier where the reference leaves it.
#
# Comparing the clouds at all requires them to be the same size: the
# reference derives its parcel count from an injection density, and the two
# have to be set to agree.
#
# The room's extreme temperature and vapour density are judged at the settled
# state and only reported over the transient.  A buoyant plume amplifies any
# difference between two runs while it is developing, so a row-by-row
# comparison of the transient measures how sensitive the plume is, not how
# closely the two couplings agree.  The vapour
# mass is an integral over the room and is comparable throughout.

include(joinpath(@__DIR__, "../../../test/caseCompare.jl"))

quantities = [
    Quantity("vapour mass, relative",
        "integralrhoV_OF.csv", "integralrhoV_ST.csv", :rel, 5e-5),
]

# Judge the last sample of a pair, where the room has settled
function check_settled!(label, ofFile, stFile, tol, unit = "")
    a = read_series(ofFile)[end]
    b = read_series(stFile)[end]
    d = maximum(abs.(a .- b))
    println()
    println("  ", label, "  (settled state)")
    println("    OF ", a)
    println("    ST ", b)
    println("    max |difference| = ", round(d, sigdigits = 6), " ", unit)
    verdict!(label, d, tol)
    return d
end

# Report a pair over the whole run without judging it
function report_transient(label, ofFile, stFile, unit = "")
    a = read_series(ofFile)
    b = read_series(stFile)
    _, d = difference(a, b, 0, :abs)
    println()
    println("  ", label, "  (transient, reported only)")
    println("    max |difference| over ", length(a), " samples = ",
        round(d, sigdigits = 6), " ", unit)
    return d
end

banner("hotRoom")
require(
    files_of(quantities)...,
    "minmaxT_OF.csv", "minmaxT_ST.csv",
    "minmaxrhoV_OF.csv", "minmaxrhoV_ST.csv",
)

for q in quantities
    check!(q)
end

check_settled!("carrier temperature",
    "minmaxT_OF.csv", "minmaxT_ST.csv", 0.05, "K")
report_transient("carrier temperature",
    "minmaxT_OF.csv", "minmaxT_ST.csv", "K")

check_settled!("carrier vapour density",
    "minmaxrhoV_OF.csv", "minmaxrhoV_ST.csv", 3e-4, "kg/m^3")
report_transient("carrier vapour density",
    "minmaxrhoV_OF.csv", "minmaxrhoV_ST.csv", "kg/m^3")

finish()
