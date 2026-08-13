#=
    SPDX-License-Identifier: GPL-3.0-or-later

    SCALE-TRACK

    Copyright (C) 2026 Sergey Lesnik

    This file is part of SCALE-TRACK, which is free software: you can
    redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.
    See <http://www.gnu.org/licenses/> for details.
=#

#=
    The comparison behind a case's Allcheck.

    A case's Allrun greps the two solver logs into paired *_OF.csv / *_ST.csv
    files; this turns those pairs into a pass/fail with an exit code.  A case
    provides only its quantities and their tolerances, in its own check
    script.

    Two things every such comparison must account for, and does here:

    - The coupling is asynchronous, so the tracking trails the synchronous
      reference and reports one sample fewer.  Every quantity is scanned over
      a range of row shifts and judged at its best; comparing unshifted
      overstates the disagreement.

    - Tolerances are observed values, not derived bounds.  They are
      regression guards: they say the case still behaves as it did, not that
      the physics is right.

    A case with a quantity that needs more than a shift scan -- a series that
    is only comparable over part of its range, say -- calls read_series and
    difference directly and hands the outcome to verdict!.
=#

# A quantity to compare.  ofcol/stcol pick one number out of a line when the
# two solvers report differently many, as for a vector against its component.
struct Quantity
    label::String
    ofFile::String
    stFile::String
    mode::Symbol        # :abs or :rel
    tol::Float64
    unit::String
    ofcol::Union{Int, Symbol}
    stcol::Union{Int, Symbol}
end

function Quantity(
    label, ofFile, stFile, mode, tol, unit = "";
    ofcol = :all, stcol = :all
)
    Quantity(label, ofFile, stFile, mode, tol, unit, ofcol, stcol)
end

# label => (value, tolerance, passed), in the order they were recorded
const VERDICTS = Tuple{String, Float64, Float64, Bool}[]

verdict!(label, value, tol) =
    push!(VERDICTS, (label, Float64(value), Float64(tol), value <= tol))

# The numbers on a grepped log line, after the "...:" or "...=" that names
# them.  Covers "T min/max: 283.3/293.0", "Temperature min/max = a, b",
# "Current mass in system = x" and "momentum [kg.m/s]: (0 0 z)" alike.
const FLOATRE = r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"

function numbers(line)
    i = findfirst(c -> c == ':' || c == '=', line)
    s = i === nothing ? line : line[nextind(line, i):end]
    return [parse(Float64, m.match) for m in eachmatch(FLOATRE, s)]
end

# One row of numbers per line, optionally reduced to a single column
function read_series(path; col = :all)
    rows = Vector{Float64}[]
    for line in eachline(path)
        v = numbers(line)
        isempty(v) && continue
        push!(rows, col === :all ? v : [v[col]])
    end
    return rows
end

#=
    The largest difference between two series with the second shifted forward,
    over rows 1:last of the first.  Returns (rows compared, difference).

    :abs   the difference itself
    :rel   relative to the reference value, row by row
    :peak  relative to the largest reference value over the range, for a
           series that passes through zero, where :rel is meaningless
=#
function difference(
    a, b, shift, mode = :abs; last = typemax(Int), lastB = typemax(Int)
)
    n = 0
    d = 0.0
    peak = 0.0
    if mode === :peak
        for i in 1:min(length(a), last), x in a[i]
            abs(x) > peak && (peak = abs(x))
        end
        peak == 0 && (peak = 1.0)
    end
    # A shift can reach past the end of what is comparable in the second
    # series, which lastB bounds independently of the first
    for i in 1:min(length(a), length(b) - shift, last, lastB - shift)
        ra, rb = a[i], b[i + shift]
        for j in 1:min(length(ra), length(rb))
            x = abs(ra[j] - rb[j])
            mode === :rel && (x = ra[j] == 0 ? 0.0 : x/abs(ra[j]))
            mode === :peak && (x = x/peak)
            x > d && (d = x)
        end
        n += 1
    end
    return n, d
end

# Scan the shifts of one pair, print the table and return (best, shift)
function scan(
    a, b, mode; maxShift = 3, unit = "", last = typemax(Int),
    lastB = typemax(Int)
)
    println("    shift   rows   max |difference|")
    best = Inf
    bestShift = 0
    for s in 0:maxShift
        n, d = difference(a, b, s, mode; last = last, lastB = lastB)
        println("    ", lpad(s, 5), " ", lpad(n, 6), " ",
            lpad(round(d, sigdigits = 6), 18), " ", unit)
        if d < best
            best = d
            bestShift = s
        end
    end
    println("    best: shift ", bestShift, " at ",
        round(best, sigdigits = 6), " ", unit)
    return best, bestShift
end

# Read a quantity's pair, scan it and record the verdict
function check!(q::Quantity; maxShift = 3)
    a = read_series(q.ofFile; col = q.ofcol)
    b = read_series(q.stFile; col = q.stcol)
    println()
    println("  ", q.label, "  (OF ", length(a), " rows / ST ", length(b), ")")
    best, shift = scan(a, b, q.mode; maxShift = maxShift, unit = q.unit)
    verdict!(q.label, best, q.tol)
    return best, shift
end

banner(title) = begin
    println()
    println(title, " -- Julia-coupled vs OpenFOAM reference")
    println("="^70)
end

# Print the recorded verdicts and exit non-zero if any failed, so a run can
# be gated on the exit status
function finish()
    println()
    println("  verdict")
    failed = 0
    for (label, value, tol, ok) in VERDICTS
        println("    ", rpad(label, 34), rpad(round(value, sigdigits = 6), 14),
            "tol ", rpad(tol, 10), ok ? "PASS" : "FAIL")
        ok || (failed += 1)
    end
    println()
    println(failed == 0 ? "  ALL CHECKS PASSED" : "  CHECKS FAILED")
    exit(failed == 0 ? 0 : 1)
end

# Every pair a case names must exist and carry rows; Allclean removes them
function require(files...)
    missing = [f for f in files if !isfile(f) || filesize(f) == 0]
    isempty(missing) && return nothing
    for f in missing
        println("  MISSING or empty: ", f)
    end
    println()
    println("  Nothing to compare.  Run ./Allrun first",
        " (Allclean removes the CSVs).")
    exit(2)
end

files_of(qs) = collect(Iterators.flatten((q.ofFile, q.stFile) for q in qs))
