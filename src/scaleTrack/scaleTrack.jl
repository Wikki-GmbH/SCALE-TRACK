#=
    SCALE-TRACK

    SCALE-TRACK is advancing two-way coupled Euler-Lagrange particle tracking
    to the realm of exascale computing enabling the simulation of dispersed
    multiphase flows at lower cost and energy consumption.  This is achieved
    through a coupling algorithm, which eliminates synchronisation barriers,
    and new cache-friendly data structures.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
    Copyright (C) 2025 Silvio Schmalfuß

    This file is part of SCALE-TRACK.

    SCALE-TRACK is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    SCALE-TRACK is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along
    with SCALE-TRACK.  If not, see <http://www.gnu.org/licenses/>.
=#

#=
    Entry point of the SCALE-TRACK particle tracking library.

    The library is include-based: a case script includes this file into Main,
    constructs a physics model -- StokesFlow or HumidAirDroplet -- and calls
    one of the two drivers,

        init_async_tracking!(executor, model; caseParameters...)
        init_sync_tracking!(executor, model; caseParameters...)

    The scalar and label type aliases must match the OpenFOAM build
    (WM_PRECISION_OPTION, WM_LABEL_SIZE).  A case script may override the
    defaults by defining the constants before including this file:

        const scalar = Float64   # DP build; default is Float32 (SP)
        const label = Int32
        include("<path to>/src/scaleTrack/scaleTrack.jl")

    See README.md for the description of the individual library files.
=#

tNow = time()

# Registry for instrumentation, caches and bookkeeping that must survive
# re-includes in an interactive session
if !isdefined(Main, :reg)
    const reg = Dict()
end
reg["gc_num"] = Base.gc_num()

using Random
# using Distributions  # Hangs when profiling with Nsight
using WriteVTK
using CUDA
using StaticArrays
using BenchmarkTools
import Adapt
import Base: *, Event
using Accessors
using MPI
using Base.Threads

ΔtLoadModules = time() - tNow

# Type aliases matching the OpenFOAM build.  Overridable by the case script
# (see the usage note above).
if !isdefined(Main, :scalar)
    const scalar = Float32
end
if !isdefined(Main, :label)
    const label = Int32
end

include("util.jl")
include("mpiUtils.jl")
include("types.jl")
include("mesh.jl")
include("control.jl")
include("comm.jl")
include("particles.jl")
include("stokesFlow.jl")
include("humidAir.jl")
include("executorCPU.jl")
include("executorGPU.jl")
include("asyncTracking.jl")
include("vtkOutput.jl")
include("coupling.jl")

# The load timings are printed by the init_*_tracking! drivers: they run
# after initComm, which silences the non-master ranks
reg["ΔtLoadModules"] = ΔtLoadModules
reg["ΔtInitMethods"] = time() - ΔtLoadModules - tNow
