#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# The executor: which device the tracking runs on, and the operations on
# Eulerian field sets that differ with it.  The executors themselves are in
# executorCPU.jl and executorGPU.jl; this file carries only what they and the
# rest of the library dispatch on.

# The tags every backend-specific method dispatches on.  GPU is abstract:
# everything that is the same for every vendor dispatches on it, while the
# concrete subtypes select the vendor API.  The selected backend defines GPU()
# to return its own subtype, so a case script keeps selecting the GPU with
# `executor = GPU()` whichever vendor it was built against.
struct CPU end
abstract type GPU end
struct CUDAGPU <: GPU end
struct ROCmGPU <: GPU end

# Helper to copy all struct data from host to device
function copy_fields!(a, b)
    for n in fieldnames(typeof(a))
        if !(typeof(getfield(b, n)) <: Number)
            if (typeof(getfield(b, n)) <: AbstractArray)
                copyto!(getfield(a, n), getfield(b, n))
            else
                copy_fields!(getfield(a, n), getfield(b, n))
            end
        end
    end
    return nothing
end

# Reset all source-term fields of an Eulerian container to zero
function reset_sources!(e)
    for f in source_fields(typeof(e))
        arr = getfield(e, f)
        fill!(arr, zero(eltype(arr)))
    end
    return nothing
end

# A task-private view of an Eulerian container: the carrier fields are shared
# for reading, the source fields are freshly allocated private accumulators
function task_view(e::E) where {E}
    E(
        e.N,
        (getfield(e, f) for f in carrier_fields(E))...,
        (similar(getfield(e, f)) for f in source_fields(E))...
    )
end
