#=
    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

#=
    Checks the Hilbert curve used to place the cloud.

    Two properties define it, and both are what the particle initialization
    relies on:

    - it visits every point of the cube exactly once, so equal stretches of
      the curve hold equal numbers of cells and the chunks come out the same
      size;
    - consecutive points are neighbours, so a chunk's stretch is a connected
      region rather than scattered cells.

    Checked exhaustively over the whole cube, which is why the resolutions
    stay small -- the transform treats every bit plane alike, so a wrong one
    shows up early.  The check also covers the mapping from a stretch of the
    curve to chunks: the stretches must tile the cube without gaps or
    overlap.

    Needs neither MPI, nor a GPU, nor OpenFOAM.

    Usage:
        julia --project=<repo root> hilbertCheck.jl
=#

const scalar = Float64
const label = Int32

include(joinpath(@__DIR__, "../../src/scaleTrack/scaleTrack.jl"))

# ------------------------------------------------------------------- the cube
const NBITS = 1:5           # 8 to 32768 points; exhaustive at each

failures = String[]

report(ok, what) =
    (println("  ", rpad(what, 52), ok ? "PASS" : "FAIL");
     ok || push!(failures, what))

println("="^78)
println("Hilbert curve")
println("="^78)

for nBits in NBITS
    nPoints = curve_length(nBits)
    side = 1 << nBits
    points = [hilbert_point(i, nBits) for i in UInt64(0):(nPoints - 1)]

    report(
        length(Set(points)) == nPoints,
        "$(side)^3: visits every point exactly once"
    )
    report(
        all(p -> all(c -> 0 <= c < side, p), points),
        "$(side)^3: stays inside the cube"
    )

    # Manhattan distance of 1 between consecutive points
    steps = [
        sum(abs.(Int.(points[i + 1]) .- Int.(points[i])))
        for i in 1:(nPoints - 1)
    ]
    report(
        all(==(1), steps),
        "$(side)^3: consecutive points are neighbours"
    )
end

# ------------------------------------------------------- the split by chunks
# The stretch a chunk gets is [(i-1)*l/n, i*l/n); the stretches have to tile
# the curve, which is what makes the chunks equal sized and disjoint.
println()
println("Chunk stretches of the curve")
println("="^78)

for nBits in 2:4, nChunks in (1, 2, 3, 5, 8)
    lCurve = curve_length(nBits)
    ranges = [chunk_curve_range(i, nChunks, nBits) for i in 1:nChunks]
    covered = union((Set(first(r):last(r)) for r in ranges)...)
    lengths = [Int(last(r)) - Int(first(r)) + 1 for r in ranges]

    report(
        length(covered) == lCurve && sum(lengths) == lCurve,
        "$(1 << nBits)^3 over $nChunks chunks: tiles the curve"
    )
    report(
        maximum(lengths) - minimum(lengths) <= 1,
        "$(1 << nBits)^3 over $nChunks chunks: equal sized to one point"
    )
end

println()
println("="^78)
if isempty(failures)
    println("VERDICT: PASS")
else
    println("VERDICT: FAIL")
    for f in failures
        println("  ", f)
    end
end
println("="^78)

# The exit status gates a run, so the test driver can read it
exit(isempty(failures) ? 0 : 1)
