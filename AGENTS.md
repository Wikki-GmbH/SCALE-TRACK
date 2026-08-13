# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

SCALE-TRACK: asynchronous two-way coupled Euler-Lagrange particle tracking.
The continuous phase is solved by OpenFOAM on CPUs; the dispersed phase is
tracked in Julia, on GPUs or on the host. The coupling removes the
synchronisation barrier between them by extrapolating the source terms and by
decomposing the Lagrangian phase independently of the Eulerian one.

Two languages meet by pointer, which is the fact most likely to catch you out —
see *Precision* below.

## Layout

| Path | |
|---|---|
| `src/scaleTrack/` | the tracking library, in Julia |
| `src/lagrangianIntermediateOpenFOAM/` | a fork of OpenFOAM's intermediate Lagrangian library; treat as vendored |
| `sol/` | solvers in pairs: `*STParcelFoam*` couples to the library, `*ParcelFoam*` is the pure-OpenFOAM reference |
| `run/` | cases, each with `Allrun` and `Allcheck` |
| `test/` | the suite, which needs neither OpenFOAM nor a GPU, and one GPU check run by hand |

The solvers come in pairs deliberately. Every claim about a coupled result is
made against the same physics solved synchronously in one stack, so a
disagreement is attributable. Do not remove a reference solver or its case to
simplify something.

`src/scaleTrack/README.md` maps the library file by file and says what a
physics model has to provide; the two shipped models are the worked examples.

## Building and testing

```sh
./Allwmake                          # library first, then the solvers
julia --project=. test/runtests.jl  # a few minutes, no OpenFOAM, no GPU
julia -e 'using JuliaFormatter; format(".")'
```

Run the suite and the formatter before proposing a change. Both are checked by
CI. The coupled comparisons against the reference solver need a built OpenFOAM
and, for the production executor, a device; they are not runnable everywhere,
so say plainly when a change has not been exercised against them.

## Precision

`scalar` and `label` in the library must match the OpenFOAM build — the coupled
fields are shared by pointer, so a mismatch corrupts data rather than failing
to link or raising an error. The default is single precision with 32-bit
labels. Never change one side alone.

## Conventions

These are enforced by review, and the history was once rewritten to bring it
into line with them.

**Comments** explain the line or block they sit on. Where a comment must point
at other code, it names the concept, not a file, a function, a call chain or a
dictionary key. Do not write into a comment:

- what a bug caused, or how the code used to behave — that is commit-message
  material
- which callers do or do not reach a path
- numbers from a measurement or an investigation; they are stale after the
  next change

A comment that would stop being true if the code it *refers to* were rewritten
does not belong there.

**Commit messages** say what changed and why. The code says how, the comments
say what a line does. Length follows the size of the change: a one-line edit
gets a subject and a sentence, reworking several files earns a paragraph. Use
list items when there is genuinely a list, never to pad. Quote a measurement
only when it is the point of the commit or changes a conclusion — a number
that sits inside its own tolerance is noise. Say where a change was verified
when that is not self-evident, or state that it was not verified at all.
Subjects fit in 72 characters, bodies wrap at 79.

**New files** carry the current year in their copyright header, not the range
an older neighbour carries.

## What not to commit

Output of a run: `dataVTK/`, `particleTimeSeries.pvd`, `stats_np*`,
`samples_np*`, `processor[0-9]*/`, `log.*`, and the `0/` and
`constant/polyMesh/` directories that the case scripts generate.

Stage explicitly. Do not use `git add -A` in a case or repository directory
that has been run in — it sweeps up exactly the above.

Not every `checks_*.txt` is a golden file. Some are the paired *outputs* of a
self-comparison and must stay untracked; check before adding one.

## Working style

- Prefer changing the library over changing a case: a fix in a case reaches
  one case and leaves the others behind. That asymmetry is why the tracking
  code was centralised in the first place.
- Both physics models go through one evolve skeleton so they cannot drift
  apart. Adding a per-model branch to shared code is usually the wrong move;
  add a model hook instead.
- The library is included, not imported, and lives in `Main` because the
  C-callable function pointers the solver fetches have to be reachable there.
  Keep that route working.
- When a change is not verifiable in the environment you are in, say so
  explicitly rather than implying it passed.
