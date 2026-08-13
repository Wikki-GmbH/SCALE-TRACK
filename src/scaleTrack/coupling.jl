#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    Case- and C++-facing API.

    A case script includes scaleTrack.jl, constructs a physics model
    (StokesFlow or HumidAirDroplet) and calls one of the two drivers:

    - init_async_tracking!(executor, model; ...): the production coupling.
      The tracking runs asynchronously with respect to the OpenFOAM time
      loop; every MPI rank participates.

    - init_sync_tracking!(executor, model; ...): a synchronous, single-rank
      variant in which evolve_cloud blocks until the tracking of the time
      step is complete.  Used for regression testing.

    Both set up the global state referenced by the C++ solver: the exported
    function pointers evolve_cloud_ptr and allocate_array_ptr as well as the
    globals comm, chunks and executor used in the solver's eval strings.

    Zero-copy field sharing: allocate_array_j hands the pointer of a
    Julia-allocated coupled field to OpenFOAM, which re-points the field's
    internal storage to it.  The coupled fields are the carrier and source
    fields of the model's Eulerian container, looked up by name.
=#

# Tags distinguishing the two coupling drivers; the active one is stored in
# the global trackingMode
struct AsyncMode end
struct SyncMode end

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

# Trigger the garbage collection manually at a fixed time step interval.
# At this point of the coupling cycle the tracking tasks are typically
# parked, so the stop-the-world pause does not interrupt an ongoing
# evolve.  Collections at any other moment are safe as well: the solver and
# the communication setup preserve Julia's SIGSEGV handler, which the
# safepoint mechanism of the multi-threaded GC relies on.
function trigger_gc_if_due!()
    global reg
    reg["timestepsSinceLastGC"] += 1
    if reg["timestepsSinceLastGC"] >= reg["gcTimeStepInterval"]
        gcDiff = Base.GC_Diff(Base.gc_num(), reg["gc_num"])
        reg["gc_num"] = Base.gc_num()
        tNow = time()
        GC.gc()
        reg["timestepsSinceLastGC"] = 0
        timing(tNow, "Run garbage collection")
        println("Allocated since last GC: $(gcDiff.allocd/1e6) MB")
    end
    return nothing
end

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

function evolve_cloud(Δt, ::AsyncMode)
    trigger_gc_if_due!()

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

###############################################################################
# Initialization drivers called by the case scripts

# Allow mesh extents and cell counts to be given as a single number (cube) or
# per direction
expand3(v::Real) = [v, v, v]
expand3(v) = collect(v)

# Resolve the cloud size into the parcel count of one chunk.  Only tracking
# masters hold chunks, so a cloud given as a whole is split over them and
# their chunks; the count then depends on the run, not on the case alone.
function parcels_per_chunk(nParcels, nParcelsTotal, nChunks, comm)
    if (nParcels === nothing) == (nParcelsTotal === nothing)
        throw(
            ErrorException(
                string(
                    "Give the cloud size either as nParcels (per chunk) or",
                    " as nParcelsTotal (over the whole cloud) -- not both",
                    " and not neither."
                )
            )
        )
    end
    nParcelsTotal === nothing && return nParcels

    nChunksGlobal = nChunks*comm.member.nHosts
    n, remainder = divrem(nParcelsTotal, nChunksGlobal)
    if n < 1
        throw(
            ErrorException(
                string(
                    "nParcelsTotal = ", nParcelsTotal, " leaves less than",
                    " one parcel for each of the ", nChunksGlobal, " chunks"
                )
            )
        )
    end
    if remainder != 0
        println(
            "nParcelsTotal = ", nParcelsTotal, " does not split evenly over ",
            nChunksGlobal, " chunks; tracking ", n*nChunksGlobal, " parcels"
        )
    end
    return n
end

function report_startup(tNow, s)
    gcDiff = Base.GC_Diff(Base.gc_num(), reg["gc_num"])
    reg["gc_num"] = Base.gc_num()
    println("Julia allocations during startup: $(gcDiff.allocd/1e6) MB")
    println(s, " done\n")
    flush(stdout)
    flush(stderr)
    return nothing
end

#=
    Set up the asynchronous two-way coupled tracking with the given physics
    model.

    The mesh description (nCellsPerDirection, origin, ending) must be kept
    consistent with the case's mesh description.  decompositions maps the MPI
    rank count to the Lagrangian domain decomposition; the entry matching the
    run's rank count is chosen, and it must evenly divide the cell counts (it
    is independent of the Eulerian decomposition apart from the rank count).

    The cloud is given either as nParcels, the number per chunk, or as
    nParcelsTotal, the number over the whole cloud -- the latter for a case
    whose cloud is to stay the same size as the rank count varies, since the
    split over the tracking masters then follows from the run.  nChunks is
    per tracking master either way.

    initChunk!(chunk, mesh, executor, iChunk, nChunksGlobal) provides the
    initial particle distribution and defaults to uniformly random positions
    and diameters; iChunk numbers the chunk within
    the whole cloud.  nSubSteps is the number of Lagrangian sub-steps per
    coupling time step.  The source extrapolator defaults to the model's
    default_extrapolator.  saveTimingsInterval > 0 writes the coupling-step
    timings to stats_np<ranks> every that many steps.
=#
function init_async_tracking!(
    executorArg, modelArg;
    nParcels = nothing,
    nParcelsTotal = nothing,
    nChunks,
    nCellsPerDirection, origin, ending,
    decompositions,
    nSubSteps = 10,
    gcTimeStepInterval = 100,
    saveTimingsInterval = 0,
    extrapolator = nothing,
    initChunk! = init!,
)
    global trackingMode = AsyncMode()
    global executor = executorArg
    global model = modelArg
    global comm = initComm(executor)

    println("Loaded modules in ",
        round(reg["ΔtLoadModules"]; sigdigits = 4), " s")
    println("Initialized methods in ",
        round(reg["ΔtInitMethods"]; sigdigits = 4), " s")
    println("Julia active project: ", Base.active_project())
    tNow = time()

    if !haskey(decompositions, comm.size)
        throw(
            ErrorException(
                string(
                    "No Lagrangian decomposition defined for ", comm.size,
                    " ranks.  Extend the decompositions table of the case."
                )
            )
        )
    end
    decomposition = decompositions[comm.size]

    reg["gcTimeStepInterval"] = gcTimeStepInterval
    reg["timestepsSinceLastGC"] = 0
    reg["timeStep"] = 0
    init_timings!(saveTimingsInterval)

    @show model
    @show nParcels nParcelsTotal nChunks nCellsPerDirection origin ending
    @show decomposition nSubSteps gcTimeStepInterval

    global mesh = construct_mesh(
        expand3(nCellsPerDirection), expand3(origin), expand3(ending),
        decomposition
    )
    tNow = timing(tNow, "Initialized mesh")

    global control = Control(
        Locks(comm.size),
        Events(),
        make_extrapolator(extrapolator, model, prod(mesh.partitionN))
    )
    tNow = timing(tNow, "Initialized control")

    # The Eulerian fields of all partitions; the entry of the own rank is
    # shared with OpenFOAM by pointer
    E = host_eulerian_type(model)
    reg["eulerian"] = Vector{E}(undef, comm.size)

    global tStart = time()
    global totalTime = 0.0
    global firstPass = true

    # Lock own Eulerian to put async evolve on wait until it's notified by
    # call to evolve_cloud from OF
    lock(control.locks.eulerianComms[comm.jlRank])

    if comm.isHost
        nPerChunk = parcels_per_chunk(nParcels, nParcelsTotal, nChunks, comm)

        # The chunks of all masters are numbered consecutively by the order
        # of the host communicator, so an initializer can hand each chunk its
        # own region of the domain
        nChunksGlobal = nChunks*comm.member.nHosts
        iChunk₀ = nChunks*comm.member.hostRank
        @show nPerChunk nChunksGlobal

        # Built by comprehension, so that the element type is the concrete
        # chunk type of this executor and model: the evolve loop then reaches
        # a chunk without dispatching on its type at every step
        global chunks = [
            let c = allocate_chunk(comm, executor, model, nPerChunk, nSubSteps)
                init_props!(c, model)
                initChunk!(c, mesh, executor, iChunk₀ + i, nChunksGlobal)
                c
            end
            for i in 1:nChunks
        ]
        tNow = timing(tNow, "Initialized particle chunks")

        # Allocate Eulerian for all ranks on tracking masters
        for i in eachindex(reg["eulerian"])
            reg["eulerian"][i] = E(prod(mesh.partitionN))
        end

        errormonitor(
            @spawn init_async_evolve!(
                chunks, model, reg["eulerian"], mesh, control, comm, executor
            )
        )
        tNow = timing(tNow, "Initialized asynchronous evolve")
    else
        # Allocate Eulerian only for self on slaves
        reg["eulerian"][comm.jlRank] = E(prod(mesh.partitionN))

        errormonitor(
            @spawn init_async_evolve!(reg["eulerian"], control, comm, executor)
        )
    end

    report_startup(tNow, "Initialize asynchronous tracking")
    return nothing
end

#=
    Set up the synchronous tracking: single rank, no MPI, evolve_cloud blocks
    until the time step's tracking is complete.  Used for regression testing
    and standalone experiments.
=#
function init_sync_tracking!(
    executorArg, modelArg;
    nParcels,
    nCellsPerDirection, origin, ending,
    nSubSteps = 10,
    initChunk! = init!,
)
    global trackingMode = SyncMode()
    global executor = executorArg
    global model = modelArg

    println("Loaded modules in ",
        round(reg["ΔtLoadModules"]; sigdigits = 4), " s")
    tNow = time()

    @show model
    @show nParcels nCellsPerDirection origin ending nSubSteps

    global mesh = construct_mesh(
        expand3(nCellsPerDirection), expand3(origin), expand3(ending),
        (1, 1, 1)
    )

    global chunk = allocate_chunk(executor, model, nParcels, nSubSteps)
    tNow = timing(tNow, "Allocated particle chunk")
    # The single chunk is the whole cloud, so there is no numbering to hand
    # out: the initializer keeps its own defaults
    init_props!(chunk, model)
    initChunk!(chunk, mesh, executor)
    tNow = timing(tNow, "Initialized particle chunk")

    # Write initial state
    write(chunk, executor)
    tNow = timing(tNow, "Written VTK data")

    reg["eulerian"] = host_eulerian_type(model)(Int(prod(mesh.N)))

    reg["timeStep"] = 0
    global totalTime = 0.0
    global firstPass = true

    report_startup(tNow, "Initialize synchronous tracking")
    return nothing
end

###############################################################################
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
