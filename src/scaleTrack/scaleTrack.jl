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
    constructs a physics model -- StokesParticle or HumidAirDroplet -- and calls
    one of the two drivers,

        init_async_tracking!(executor, model; caseParameters...)
        init_sync_tracking!(executor, model; caseParameters...)

    The scalar and label type aliases must match the OpenFOAM build
    (WM_PRECISION_OPTION, WM_LABEL_SIZE).  A case script may override the
    defaults by defining the constants before including this file, and it
    selects the GPU vendor the same way:

        const scalar = Float64   # DP build; default is Float32 (SP)
        const label = Int32
        const gpuBackend = :ROCm # AMD GPUs; default is :CUDA
        include("<path to>/src/scaleTrack/scaleTrack.jl")

    Only the packages of the selected vendor are loaded, so a machine needs
    CUDA.jl or AMDGPU.jl, not both.  `executor = GPU()` then yields whichever
    backend was selected.  A case that does not set gpuBackend takes it from
    the environment variable ST_GPU_BACKEND, so the same case runs on either
    vendor without editing.

    Each file below carries a header comment describing what it holds.
=#

tNow = time()

# Registry for instrumentation, caches and bookkeeping that must survive
# re-includes in an interactive session
if !isdefined(Main, :reg)
    const reg = Dict()
end
reg["gc_num"] = Base.gc_num()

using Random
using WriteVTK
using StaticArrays
using BenchmarkTools
import Adapt
import Base: *, Event
using Accessors
using MPI
using Base.Threads

# The GPU vendor backend.  Only the selected one is loaded, so a machine needs
# the packages of its own vendor only.  Overridable by the case script in the
# same way as the type aliases below, or -- since the vendor is a property of
# the machine rather than of the case -- through the environment, which lets
# the same case run on either.
if !isdefined(Main, :gpuBackend)
    const gpuBackend = Symbol(get(ENV, "ST_GPU_BACKEND", "CUDA"))
end

if gpuBackend === :CUDA
    using CUDA
elseif gpuBackend === :ROCm
    using AMDGPU
else
    error("Unknown gpuBackend $(gpuBackend); expected :CUDA or :ROCm")
end

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
include("mpi.jl")
include("hilbert.jl")
include("types.jl")
include("mesh.jl")
include("control.jl")
include("ranks.jl")
include("parcels.jl")
include("stokesParticle.jl")
include("humidAirDroplet.jl")
include("executorCPU.jl")
include("executorGPU.jl")
if gpuBackend === :CUDA
    include("backendCUDA.jl")
else
    include("backendROCm.jl")
end
include("asyncTracking.jl")
include("vtkOutput.jl")
include("coupling.jl")

# The load timings are printed by the initialization drivers, which run after
# the non-master ranks have been silenced
reg["ΔtLoadModules"] = ΔtLoadModules
reg["ΔtInitMethods"] = time() - ΔtLoadModules - tNow
