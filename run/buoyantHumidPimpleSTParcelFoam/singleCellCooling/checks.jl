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

# What Allcheck compares for singleCellCooling, and how closely.
#
# One droplet cooling by convection alone into the single cell it sits in.
# The carrier temperature is the source term made visible, the parcel
# temperature the kernel.

include(joinpath(@__DIR__, "../../../test/caseCompare.jl"))

quantities = [
    Quantity("carrier temperature",
        "Tcont_OF.csv", "Tcont_ST.csv", :abs, 0.01, "K"),
    Quantity("parcel temperature", "Tp_OF.csv", "Tp_ST.csv", :abs, 0.05, "K"),
]

banner("singleCellCooling")
require(files_of(quantities)...)

for q in quantities
    check!(q)
end

finish()
