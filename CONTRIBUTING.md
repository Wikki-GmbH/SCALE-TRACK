# Contributing

## Building

The solvers embed the Julia runtime, so `julia` must be on `PATH` when they are
compiled — the build shells out to `julia-config.jl` for its flags.  An
OpenFOAM environment (openfoam.com/ESI) must be sourced.

```sh
./Allwmake
```

That builds the forked Lagrangian library first and the three solvers after
it, which is the order the dependency requires.

**The precision has to match across the language boundary.**  The solvers are
built single precision with 32-bit labels, because the tracking library
defines its scalar as `Float32` and its label as `Int32`.  Fields are shared
by pointer, so a double-precision build corrupts data rather than failing to
link.  `icoJuliaParcelFoam` also has a double-precision variant, used by the
CPU test case.

## The Julia environment

One environment serves the whole repository.  Instantiate it once from the
root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
```

The second command is not optional for a parallel run: MPI.jl has to be
pointed at the same MPI the solvers were built against.  It writes a
machine-specific `LocalPreferences.toml`, which is not tracked.

## Before you push

```sh
julia --project=. test/runtests.jl
julia -e 'using JuliaFormatter; format(".")'
```

The suite needs neither OpenFOAM nor a GPU and takes a few minutes.  Both are
run by CI on every push, so a change that fails either will not go unnoticed —
but finding out locally is faster.

The coupled comparisons against the reference solver are a separate matter:
they need a built OpenFOAM, and for the production executor a device.  Run
them on a machine that has both, with the `Allrun` and `Allcheck` of the case.

## Comments and commit messages

These rules are not decorative — the history was rewritten once to bring it
into line with them:

- A comment explains the line or block it sits on.  Where it must point at
  other code it names the concept, not a file, a function or a dictionary key.
- What a bug caused, how the code used to behave, and what a measurement
  showed belong in the commit message, not in a comment.
- A commit message says what changed and why; the code says how.  Its length
  follows the size of the change.  Say where a change was verified when that
  is not self-evident, or that it was not verified at all.
- No trailers.

## Adding a physics model

`src/scaleTrack/model.jl` states what a model has to provide and carries the
fallbacks it may leave alone.  The two shipped models are the worked examples:
`stokesParticle.jl` for the momentum-only case, `humidAirDroplet.jl` for one
that also exchanges heat and vapour mass.
