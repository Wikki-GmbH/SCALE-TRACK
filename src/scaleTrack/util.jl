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
