#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Driving the library without OpenFOAM, as in a REPL: fill the carrier fields
# with data and run a few coupling steps.

# Standalone helpers (running without OpenFOAM, e.g. in a REPL)

# Fill the carrier velocity fields with seeded random data.  Models with
# additional carrier fields (temperature, vapour density) need those
# initialized to physically sensible ranges by the caller.
function randomize_velocity!()
    e = reg["eulerian"]
    if e isa Vector
        for i in eachindex(e)
            # Slaves allocate only their own partition
            isassigned(e, i) && init_random!(e[i].U, 2SCL, -1SCL)
        end
    else
        init_random!(e.U, 2SCL, -1SCL)
    end
    return nothing
end

# Exercise the tracking without OpenFOAM: freeze a random velocity field and
# run a few evolve steps
function standalone_run!(nSteps = 2, Δt = 1e-3)
    randomize_velocity!()
    tNow = time()
    for _ in 1:nSteps
        evolve_cloud(Δt)
    end
    timing(tNow, "Standalone run of $nSteps evolve steps")
    return nothing
end
