# The tracking library

The Lagrangian phase: parcels, the physics that advances them, and the
coupling that exchanges fields with the OpenFOAM side.

## How a case uses it

A case script sets what is its own, includes this directory's entry point, and
calls a driver.

```julia
const CloudLogFrequency = 1   # optional: report the cloud every step

include(joinpath(@__DIR__, "../../../src/scaleTrack/scaleTrack.jl"))

physics = StokesParticle(μᶜ = 1e-3, ρᶜ = 1e3, ρᵈ = 1.0)

init_async_tracking!(
    GPU(), physics;
    nParcels = 100_000, nChunks = 10,
    nCellsPerDirection = 40, origin = 0.0, ending = 1.0,
    decompositions = Dict(1 => (1, 1, 1), 20 => (2, 2, 5)),
)
```

The mesh description must agree with the case's mesh, and the Lagrangian
decomposition must divide the cell counts evenly.  It is otherwise independent
of the Eulerian decomposition — only the rank count is shared.

`init_sync_tracking!` is the synchronous, single-rank counterpart, used by the
checks: it blocks until the step's tracking is done, which is what makes a
result reproducible enough to checksum.

The library is included rather than imported.  It lives in `Main`, which is
where the C-callable function pointers the solver fetches have to be, and a
case reaches it by relative path, so a checkout runs against its own sources
with no environment step in between.

## The files

| | |
|---|---|
| `scaleTrack.jl` | entry point: the type aliases, the backend selection, the include list |
| `types.jl` | numeric shorthands, small vectors, the parcel containers |
| `executor.jl` | the tags every backend-specific method dispatches on, and the field-set operations that differ with the device |
| `mesh.jl` | the structured mesh and parcel localization |
| `hilbert.jl` | the three-dimensional Hilbert curve |
| `extrapolator.jl` | what stands in for a source between the steps that produce one |
| `control.jl` | the locks and events the solver thread and the tracking task hand over on |
| `ranks.jl` | rank roles and the communicators |
| `mpiWrappers.jl` | allocation-free MPI wrappers, and the communication debug logging |
| `parcels.jl` | the model-independent evolve skeleton |
| `parcelInit.jl` | how a chunk's parcels start out |
| `cloudSummary.jl` | the cloud report and the gate that drops it when unused |
| `timings.jl` | what is measured per coupling step, and how the summary and its samples are written |
| `stokesParticle.jl` | momentum-only physics: a particle under Stokes drag |
| `humidAirDroplet.jl` | a water droplet in humid air: drag, heat and vapour mass |
| `executorCPU.jl` | tracking on the host, over the default thread pool |
| `executorGPU.jl` | tracking on a device, vendor independent |
| `backendCUDA.jl`, `backendROCm.jl` | the vendor primitives the GPU executor dispatches to |
| `asyncTracking.jl` | the asynchronous master and slave loops, and the rank-to-rank protocol |
| `api.jl` | what the embedded solver calls: the two exported function pointers |
| `coupling.jl` | what a case script calls: the two drivers |
| `standalone.jl` | driving the library without OpenFOAM, as in a REPL |
| `vtkOutput.jl` | writing parcels for ParaView |

## Adding a physics model

`stokesParticle.jl` is the interface at its smallest and `humidAirDroplet.jl`
the same interface used fully; between them they define every method the
tracking calls on a model, which is what a new one has to define too.  A model
is a value dispatched on, so what it owns is its Eulerian field set, the
per-parcel arrays it needs, the carrier state read at the parcel's cell, and
the sub-step physics with its source accumulator.  Everything else —
localizing the parcel, sub-stepping, bouncing at the boundary, flushing the
sources on a cell change — is the skeleton in `parcels.jl`, which every model
goes through, so two models cannot drift apart in the parts they share.

Three of the methods have a fallback a model may leave alone: the initial
values of its per-parcel properties, in `parcelInit.jl`, and the parcel weight
and property names the summary reduces over, in `cloudSummary.jl`.

Both shipped models take submodel type parameters, so a case that switches
evaporation or heat transfer off compiles none of it rather than multiplying
by zero.

## Precision

`scalar` and `label` must match the OpenFOAM build, since the coupled fields
are shared by pointer.  The defaults are `Float32` and `Int32`; a case built
against a double-precision OpenFOAM sets `const scalar = Float64` before the
include.

The GPU vendor is selected the same way, or from the environment — it is a
property of the machine rather than of the case, so the same case runs on
either.  Only the selected backend is loaded.
