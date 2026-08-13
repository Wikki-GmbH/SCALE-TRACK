#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    What a physics model is.

    A model is a value the case constructs and passes to a driver.  Everything
    that depends on the physics dispatches on its type, so adding one is a
    matter of giving these a method and nothing else:

    - the Eulerian field set it reads and writes, and which of those fields
      are carrier and which are source
    - the per-parcel arrays it needs beyond position, velocity and diameter,
      and the value each starts at
    - the mapping between those arrays and the immutable state a sub-step
      advances
    - the carrier state read at the parcel's cell, and the reread after the
      parcel changes cell
    - the sub-step physics, its source accumulator, the boundary-bounce
      correction and the write-back into the Eulerian fields
    - the disperse-phase density, the parcel weight, and the source
      extrapolator it defaults to

    The fallbacks below are the ones a model may leave alone.  Everything else
    it must provide, and the two shipped models are the worked examples.
=#

# Initial values of a model's per-parcel property arrays.  The drivers apply
# this to every chunk before the initializer runs, so an initializer only has
# to set what its case wants different.  Zero is a usable state for a property
# that merely accumulates; one that enters a law as a divisor or an argument
# of a nonlinear function has to name a value here, or a cloud that keeps the
# default state cannot be evolved at all.
init_props!(chunk, model) = foreach(a -> fill!(a, 0.0), values(chunk.props))

# Physical particles one tracked parcel stands for.  The summary reports the
# physical cloud, as an OpenFOAM cloud's info() does, so the extensive
# quantities are weighted by it; a model without the notion weighs 1.
parcel_weight(model) = 1SCL

# Names of the model's per-parcel property arrays, needed on ranks that hold
# no chunks to build a matching reduction buffer
prop_names(model) = keys(parcel_props(model, Vector{scalar}, 0))
