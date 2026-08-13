#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    The two drivers a case script calls.

    A case includes scaleTrack.jl, constructs a physics model -- StokesParticle
    or HumidAirDroplet -- and calls one of:

    - init_async_tracking!(executor, model; ...): the production coupling.
      The tracking runs asynchronously with respect to the OpenFOAM time
      loop; every MPI rank participates.

    - init_sync_tracking!(executor, model; ...): a synchronous, single-rank
      variant in which the coupling blocks until the tracking of the time
      step is complete.  Used for regression testing.

    Either one sets up the global state the solver reaches through: the
    communicator and its role, the mesh, the chunks, the executor, and the
    mode that decides which methods the exported entry points dispatch to.
=#

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
    write_chunk(chunk, executor)
    tNow = timing(tNow, "Written VTK data")

    reg["eulerian"] = host_eulerian_type(model)(Int(prod(mesh.N)))

    reg["timeStep"] = 0
    global totalTime = 0.0
    global firstPass = true

    report_startup(tNow, "Initialize synchronous tracking")
    return nothing
end
