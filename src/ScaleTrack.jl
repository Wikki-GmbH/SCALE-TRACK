#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack/scaleTrack.jl or <http://www.gnu.org/licenses/>
    for details.

    Copyright (C) 2026 Sergey Lesnik
    Copyright (C) 2026 Henrik Rusche
    Copyright (C) 2026 Silvio Schmalfuß
=#

#=
    The package form of the tracking library.

    The library proper is include-based and is loaded into Main by the
    embedded solver, which needs the exported function pointers to be reachable
    there.  This module is the other way in: `using ScaleTrack` for a caller
    that only wants the library, without OpenFOAM and without the globals the
    coupling keeps.

    Both load the same sources, so there is one implementation.

    The precision and the GPU vendor are compile-time constants a case sets
    before loading.  Through this module they are taken from Main if defined
    there when it is first loaded, and are fixed for the session afterwards --
    a caller wanting a different precision has to say so before the first
    `using`.  A case that needs to choose per run should include the library
    directly, as the solvers do.
=#
module ScaleTrack

include("scaleTrack/scaleTrack.jl")

# The physics models and their submodels
export StokesParticle, HumidAirDroplet
export Evaporation, NoEvaporation, HeatTransfer, NoHeatTransfer

# The executors
export CPU, GPU

# The drivers a case calls, and the entry points the solver calls
export init_async_tracking!, init_sync_tracking!
export evolve_cloud, evolve_cloud_ptr, allocate_array_ptr

# Source extrapolation
export ConstExtrapolator, NoExtrapolator

# Driving the library without OpenFOAM
export standalone_run!, randomize_velocity!

#=
    Bind the C-callable entry points into Main.

    The embedded solver fetches the two function pointers by name from Main,
    so a run that reaches the library through this module has to put them
    there.  A run that includes the library directly already has them.
=#
function install_in_main!()
    for n in (:evolve_cloud_ptr, :allocate_array_ptr)
        Core.eval(Main, :(const $n = $(getfield(ScaleTrack, n))))
    end
    return nothing
end

end # module
