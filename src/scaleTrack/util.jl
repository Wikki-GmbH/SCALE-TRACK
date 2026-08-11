#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Auxiliary methods: timing helpers and debug printing

# Debug communication logging.  If true, log to stdout from every rank
# prefixed by [rank].  May be overridden by defining the constant before
# including the library.
if !isdefined(Main, :DebugComm)
    const DebugComm = false
end

function timing(t, s)
    dt = round(time() - t, sigdigits=4)
    println(s, " in ", dt, " s")

    # Invoke flush to ensure immediate printing even when executed within C code
    flush(stdout)
    return time()
end

#=
    Coupling-step timings.

    Four times are recorded per coupling step, all of them wall clock and all
    of them measured on the rank that records them:

    step            the whole step, from one call of the coupling to the next
    euler           the part of it spent in the Eulerian solver, i.e. outside
                    the tracking
    evolve          how long the tracking took, communication included
    deviceCompute   the compute alone, without the communication around it

    Whether the tracking is actually paid for is the point of collecting
    both step and evolve: when the coupling does its job, evolve overlaps
    the Eulerian solve and the step costs no more than the solver alone.

    Note that step and euler are recorded where the solver calls in, while
    evolve and deviceCompute are recorded in the tracking task, so the
    counts of the two pairs differ by whatever is in flight -- the summary
    truncates them to a common length.
=#

# Steps left out of the statistics.  The first ones carry the compilation of
# the kernels and the first-touch of the buffers, which are startup costs
# rather than the per-step cost being measured.
const nSkipTimingSteps = 2

const timingNames = ("step", "euler", "evolve", "deviceCompute")

function init_timings!(writeInterval)
    reg["timingsWriteInterval"] = writeInterval
    reg["tStepStart"] = time()
    reg["tEulerStart"] = time()
    for n in timingNames
        reg["dt_"*n] = Float64[]
    end
    return nothing
end

# Record one sample, unless the step is still one of the skipped ones
function record_timing!(name, dt, iStep)
    iStep > nSkipTimingSteps && push!(reg["dt_"*name], dt)
    return nothing
end

sample_mean(v) = isempty(v) ? 0.0 : sum(v)/length(v)

function sample_std(v)
    length(v) < 2 && return 0.0
    m = sample_mean(v)
    return sqrt(sum(x -> (x - m)^2, v)/(length(v) - 1))
end

# Write the timing summary of the run so far.  One row, so that the files of
# a scaling series concatenate into a table.
function save_timings(comm)
    # The counts differ between the pairs, so compare like with like
    nSteps = minimum(length(reg["dt_"*n]) for n in timingNames)
    series = [first(reg["dt_"*n], nSteps) for n in timingNames]

    open("stats_np" * lpad(string(comm.size), 4, '0'), "w") do io
        println(io, join(
            ["nTimeSteps";
             ["t" * uppercasefirst(n) * s
              for s in ("Total", "Mean", "Std") for n in timingNames]
            ], " "
        ))
        print(io, nSteps)
        for f in (sum, sample_mean, sample_std), v in series
            print(io, " ", round(f(v), sigdigits=6))
        end
        println(io)
    end
    return nothing
end

timings_write_due(iStep) =
    reg["timingsWriteInterval"] > 0 &&
    iStep > nSkipTimingSteps &&
    iStep % reg["timingsWriteInterval"] == 0

# Macro for printing debug statements.  Enable by setting global DebugComm to
# true.  If disabled, all debug printing is turned off with zero overhead.
macro debugCommPrintln(ex)
    if DebugComm
        # Put message into a single string before printing to avoid output
        # overlap
        msg = :(Main.Base.inferencebarrier(Main.Base.string)(
            "[", comm.rank, "] ", $(esc(ex)), "\n"
        ))
        return :( print($msg) )
    end
    return nothing
end

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

# Whether this coupling step is a reporting one.  Only ever called from
# within @cloudSummary, so the frequency is known to be positive.
cloud_summary_due(step) = (step % CloudLogFrequency == 0)
