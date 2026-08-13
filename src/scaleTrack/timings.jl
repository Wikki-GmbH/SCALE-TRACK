#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Timing of the coupling.  What is measured, where each series is taken,
# and how the run's summary and its samples are written out.

function timing(t, s)
    dt = round(time() - t, sigdigits = 4)
    println(s, " in ", dt, " s")

    # Invoke flush to ensure immediate printing even when executed within C code
    flush(stdout)
    return time()
end

# Steps left out of the statistics.  The first ones carry the compilation of
# the kernels and the first-touch of the buffers, which are startup costs
# rather than the per-step cost being measured.
const nSkipTimingSteps = 2

#=
    The times recorded per coupling step, all of them wall clock and all of
    them measured on the rank that records them.

    Three are taken where the solver calls into the coupling:

    step            the whole step, from one call of the coupling to the next
    euler           the part of it spent in the Eulerian solver, i.e. outside
                    the tracking
    wait            the rest of it: what the solver spends blocked in the
                    coupling, waiting for the tracking of the step to reach
                    the point where the fields may be exchanged

    so that step = euler + wait, and the Eulerian phase is separable from the
    synchronization rather than inferred from it.

    The others are taken in the tracking task, and are the phases of one
    evolve in the order they run.  deviceCompute is the compute alone, and it
    is the only phase that can overlap the Eulerian solve -- the solver cannot
    return until the carrier fields of the next evolve are negotiated and
    copied, and it cannot pass its next call until the sources of this one are
    copied and exchanged.
    waitEuler is the mirror of wait, the tracking waiting for the solver,
    which is where a coupling that overlaps well spends its time.

    The two groups are recorded in different tasks, so their counts differ by
    whatever is in flight; the summary truncates them to a common length.
=#
const timingNames = (
    "step", "euler", "wait", "deviceCompute",
    "negotiate", "copyCarrier", "waitEuler", "copySource", "exchangeSource"
)

function init_timings!(writeInterval)
    reg["timingsWriteInterval"] = writeInterval
    reg["tStepStart"] = time()
    reg["tEulerStart"] = time()
    for n in timingNames
        reg["dt_" * n] = Float64[]
    end
    return nothing
end

# Record one sample, unless the step is still one of the skipped ones
function record_timing!(name, dt, iStep)
    iStep > nSkipTimingSteps && push!(reg["dt_" * name], dt)
    return nothing
end

# One line per step on what the tracking cost and what the solver waited for
# it, from the series rather than from the tracking task itself
function report_evolve()
    dtCompute = reg["dt_deviceCompute"]
    dtWait = reg["dt_wait"]
    (isempty(dtCompute) || isempty(dtWait)) && return nothing
    println("Lagrangian solver: compute = ",
        round(dtCompute[end], sigdigits = 4), " s; solver waited = ",
        round(dtWait[end], sigdigits = 4), " s")
    flush(stdout)
    return nothing
end

sample_mean(v) = isempty(v) ? 0.0 : sum(v)/length(v)

function sample_std(v)
    length(v) < 2 && return 0.0
    m = sample_mean(v)
    return sqrt(sum(x -> (x - m)^2, v)/(length(v) - 1))
end

#=
    Write the samples the summary is made of, one row per coupling step.

    The summary alone cannot say how well a mean is resolved.  A coupling
    whose tracking outlasts the Eulerian solve alternates -- the solver waits
    for a whole tracking on one step and for nothing on the next -- and the
    standard deviation of such a series measures the alternation, not the
    precision of its mean.  That needs the samples: a confidence interval
    taken over batches of steps, the batch a multiple of the period.
=#
function save_samples(comm, series, nSteps)
    open("samples_np" * lpad(string(comm.size), 4, '0'), "w") do io
        println(io, join(["iStep"; collect(timingNames)], " "))
        for i in 1:nSteps
            println(
                io,
                join(
                    [i + nSkipTimingSteps;
                        [round(v[i], sigdigits = 6) for v in series]], " "
                )
            )
        end
    end
    return nothing
end

# Write the timing summary of the run so far.  One row, so that the files of
# a scaling series concatenate into a table.
function save_timings(comm)
    # The counts differ between the pairs, so compare like with like
    nSteps = minimum(length(reg["dt_" * n]) for n in timingNames)
    series = [first(reg["dt_" * n], nSteps) for n in timingNames]
    save_samples(comm, series, nSteps)

    open("stats_np" * lpad(string(comm.size), 4, '0'), "w") do io
        println(
            io,
            join(
                ["nTimeSteps";
                    [
                        "t" * uppercasefirst(n) * s
                        for s in ("Total", "Mean", "Std") for n in timingNames
                    ]
                ], " "
            )
        )
        print(io, nSteps)
        for f in (sum, sample_mean, sample_std), v in series
            print(io, " ", round(f(v), sigdigits = 6))
        end
        println(io)
    end
    return nothing
end

timings_write_due(iStep) =
    reg["timingsWriteInterval"] > 0 &&
    iStep > nSkipTimingSteps &&
    iStep % reg["timingsWriteInterval"] == 0
