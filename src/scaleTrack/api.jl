#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# The interface the embedded solver drives the tracking through.
#
# Two function pointers are exported as C callables: one the solver calls each
# time step to advance the cloud, one it calls at start-up to have the coupled
# fields allocated on the Julia side, whose storage it then re-points its own
# fields at.  Both dispatch on the mode the active driver set.

# The Eulerian container whose fields are shared with OpenFOAM on this rank
coupled_eulerian(::AsyncMode) = reg["eulerian"][comm.jlRank]
coupled_eulerian(::SyncMode) = reg["eulerian"]

coupled_field_size(::AsyncMode) = Int(prod(mesh.partitionN))
coupled_field_size(::SyncMode) = Int(prod(mesh.N))

function allocate_array_j(
    name::Cstring, size::Cint, nComponents::Cint, typeByteSize::Cint
)::Ptr{Cdouble}
    GC.@preserve name nameSymbol = Symbol(unsafe_string(pointer(name)))
    print("Allocating $nameSymbol\n")

    if Int(typeByteSize) != Int(sizeof(scalar))
        throw(
            ErrorException(
                string(
                    "Size of Julia type (in Bytes): ", sizeof(scalar),
                    " is different from the extern type: ", typeByteSize
                )
            )
        )
    end

    # Called through a @cfunction pointer, whose world age predates any
    # method added after the library include, so dispatch in the newest world
    expectedSize = Base.invokelatest(coupled_field_size, trackingMode)
    if Int(size) != expectedSize
        throw(
            ErrorException(
                string(
                    "Size of Julia mesh: ", expectedSize,
                    " is different from field's ", nameSymbol, " size: ",
                    Int(size)
                )
            )
        )
    end

    # By now, assume that the requested field already has been allocated
    field = getfield(
        Base.invokelatest(coupled_eulerian, trackingMode), nameSymbol
    )
    GC.@preserve field cFieldPtr = Base.unsafe_convert(Ptr{Cdouble}, field)
    print("Allocating $nameSymbol done\n")
    return cFieldPtr
end

const allocate_array_ptr =
    @cfunction(allocate_array_j, Ptr{Cdouble}, (Cstring, Cint, Cint, Cint))

function evolve_cloud(Δt, ::AsyncMode)
    reg["timeStep"] += 1
    iStep = reg["timeStep"]

    # The solver calls in once per step and always at the same point of its
    # loop, so one step ends where the next begins; what is not the tracking
    # is the Eulerian solver
    tNow = time()
    record_timing!("step", tNow - reg["tStepStart"], iStep)
    record_timing!("euler", tNow - reg["tEulerStart"], iStep)
    reg["tStepStart"] = tNow

    # Reported before the next evolve is unblocked, which is where the chunks
    # are settled: they then hold the state of the previous coupling step, so
    # the step just completed is the one indexed here.  The report trails by
    # the one step the coupling is asynchronous by.
    @cloudSummary begin
        completed = iStep - 1
        completed > 0 && cloud_summary_due(completed) &&
            report_cloud_summary(AsyncMode())
    end

    if comm.isMaster
        print("Evolve cloud\n")
        for chunk in chunks
            increment_time!(chunk, Δt, comm, executor)
        end
    end

    evolve!(control, executor)

    # Reported here rather than by the tracking task, whose writes to a
    # redirected stdout would not complete until this thread came back.  The
    # tracking figure is the last one it recorded, so it trails this step by
    # whatever the coupling has in flight.
    comm.isMaster && report_evolve()

    comm.isMaster && timings_write_due(iStep) && save_timings(comm)
    reg["tEulerStart"] = time()

    return nothing
end

function evolve_cloud(Δt, ::SyncMode)
    println("Evolve particles")
    flush(stdout)

    tNow = time()

    reset_sources!(reg["eulerian"])

    sync_evolve!(chunk, model, reg["eulerian"], mesh, Δt, executor)

    @cloudSummary begin
        reg["timeStep"] += 1
        cloud_summary_due(reg["timeStep"]) && report_cloud_summary(SyncMode())
    end

    tEvolve = time() - tNow
    if !firstPass
        global totalTime += tEvolve
    end
    global firstPass = false
    println("Lagrangian solver timings: current evolve = ",
        round(tEvolve, sigdigits = 4), " s; total time = ",
        round(totalTime, sigdigits = 4), " s"
    )
    return nothing
end

# Called by the solver each time step; dispatches to the active driver.
# The @cfunction trampoline below captures the world age of its creation and
# would not see methods defined after it; invokelatest dispatches in the
# newest world, so the mode methods may be (re)defined at any point, e.g. by
# a case script.
function evolve_cloud(Δt)
    Base.invokelatest(evolve_cloud, Δt, trackingMode)
    return nothing
end

const evolve_cloud_ptr = @cfunction(evolve_cloud, Cvoid, (Cdouble,))
