#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Coupling steps between two cloud summaries; 0 disables them.  Matches the
# reporting interval a Lagrangian cloud solution offers.  May be overridden
# by defining the constant before including the library.
if !isdefined(Main, :CloudLogFrequency)
    const CloudLogFrequency = 0
end

# Macro guarding the cloud summary.  At a frequency of 0 the expression is
# dropped at parse time, so a run that does not ask for the summary carries
# neither the reductions nor a branch over them.
macro cloudSummary(ex)
    CloudLogFrequency > 0 ? esc(ex) : nothing
end

# Whether this coupling step is a reporting one.  The frequency is positive
# here; a zero frequency is dropped before this is reached.
cloud_summary_due(step) = (step % CloudLogFrequency == 0)

# Cloud summary: the parcel state after a coupling step, in the shape a
# Lagrangian solver reports it.  Guarded, so a run that does not ask for the
# summary compiles none of this.

# Physical particles one tracked parcel stands for.  The summary reports the
# physical cloud, as an OpenFOAM cloud's info() does, so the extensive
# quantities are weighted by it; a model without the notion weighs 1.
parcel_weight(model) = 1SCL

# Local extrema and momentum of one chunk.  The momentum reduction fuses into
# one broadcast, so it costs a single temporary of the chunk's length -- one
# of the reasons the summary is opt-in and rate limited.
function chunk_summary(chunk, model)
    c = chunk
    dMin, dMax = extrema(c.d)
    w = parcel_weight(model)*parcel_density(model)*π/6
    mass = w*sum(c.d .^ 3)
    momentum = w*sum(c.d .^ 3 .* c.W)
    props = map(a -> extrema(a), values(c.props))
    return (N = Int(c.N), dMin = dMin, dMax = dMax, mass = mass,
        momentum = momentum, props = props)
end

# Combine the per-chunk summaries of this rank
function combine_summaries(summaries)
    first = summaries[1]
    N = sum(s -> s.N, summaries)
    dMin = minimum(s -> s.dMin, summaries)
    dMax = maximum(s -> s.dMax, summaries)
    mass = sum(s -> s.mass, summaries)
    momentum = sum(s -> s.momentum, summaries)
    props = ntuple(length(first.props)) do i
        (minimum(s -> s.props[i][1], summaries),
            maximum(s -> s.props[i][2], summaries))
    end
    return (N = N, dMin = dMin, dMax = dMax, mass = mass,
        momentum = momentum, props = props)
end

# Print in the layout of an OpenFOAM cloud info() block
function print_cloud_summary(s, propNames)
    println("Cloud summary")
    println("    Current number of parcels   = ", s.N)
    println("    Current mass in system      = ", s.mass)
    println("    Diameter min/max            = ", s.dMin, ", ", s.dMax)
    for (name, ex) in zip(propNames, s.props)
        println("    ", rpad(string(name), 8), " min/max            = ",
            ex[1], ", ", ex[2])
    end
    println("    Linear momentum z           = ", s.momentum)
    flush(stdout)
    return nothing
end

# Names of the model's per-parcel property arrays, needed on ranks that hold
# no chunks to build a matching reduction buffer
prop_names(model) = keys(parcel_props(model, Vector{scalar}, 0))

# Report the parcel state of the whole cloud.  Every rank contributes -- those
# without chunks contribute neutral elements -- and the reduced values are
# printed once, so the numbers are comparable with the reference cloud's.
@cloudSummary function report_cloud_summary(::AsyncMode)
    names = prop_names(model)
    nP = length(names)

    sums = zeros(Float64, 3)            # parcel count, mass, linear momentum
    mins = fill(Inf, 1 + nP)            # diameter, then the model's props
    maxs = fill(-Inf, 1 + nP)

    if comm.isHost
        s = combine_summaries([chunk_summary(c, model) for c in chunks])
        sums[1] = s.N
        sums[2] = s.mass
        sums[3] = s.momentum
        mins[1] = s.dMin
        maxs[1] = s.dMax
        for i in 1:nP
            mins[1 + i] = s.props[i][1]
            maxs[1 + i] = s.props[i][2]
        end
    end

    MPI.Allreduce!(sums, MPI.SUM, comm.communicator)
    MPI.Allreduce!(mins, MPI.MIN, comm.communicator)
    MPI.Allreduce!(maxs, MPI.MAX, comm.communicator)

    comm.isMaster || return nothing
    print_cloud_summary(
        (
            N = round(Int, sums[1]),
            dMin = mins[1],
            dMax = maxs[1],
            mass = sums[2],
            momentum = sums[3],
            props = ntuple(i -> (mins[1 + i], maxs[1 + i]), nP),
        ),
        names
    )
    return nothing
end

@cloudSummary function report_cloud_summary(::SyncMode)
    print_cloud_summary(chunk_summary(chunk, model), keys(chunk.props))
    return nothing
end
