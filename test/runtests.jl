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
    The checks that need neither OpenFOAM nor a GPU, run as one suite.

        julia --project=. test/runtests.jl

    Every check runs in its own process.  It has to: the library fixes
    `scalar` and `label` as constants when it is included, so a precision is
    chosen once per process, and it defines everything in Main, which a
    second include would collide with.  The driver therefore reports exit
    statuses rather than calling into the checks.

    Outputs are written to a temporary directory.  Two of the checks compare
    files they generate, and those files are not to be committed -- writing
    them outside the tree is what keeps that from happening by accident.

    The coupled comparison against the OpenFOAM reference solver is a
    separate matter and is not run here.
=#

using Test

const ROOT = abspath(joinpath(@__DIR__, ".."))
const JULIA = Base.julia_cmd()

# Run a check as its own process, silently unless it fails or
# SCALETRACK_TEST_VERBOSE is set.  Passing quiet = true suppresses the output
# of a check whose failure is expected, so the suite stays readable.
# Returns true if it exited zero.
function run_check(script, args...; threads = nothing, quiet = false)
    cmd = `$JULIA --project=$ROOT`
    threads === nothing || (cmd = `$JULIA --project=$ROOT --threads=$threads`)
    cmd = `$cmd $(joinpath(ROOT, script)) $(collect(args))`

    out = IOBuffer()
    ok = success(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    if (!ok && !quiet) || haskey(ENV, "SCALETRACK_TEST_VERBOSE")
        println(String(take!(out)))
    end
    return ok
end

# Read a checksum file as name => rest-of-line
function read_checks(path)
    d = Dict{String, String}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        name, rest = split(line, limit = 2)
        d[name] = strip(rest)
    end
    return d
end

mktempdir() do tmp

@testset "scaleTrack" begin

    # --------------------------------------------------------- verification
    # Against analytic references and against each other, so a failure means
    # the physics or the driver is wrong, not that it changed.
    @testset "single droplet against RK4" begin
        @test run_check("test/scaleTrack/singleDropletCheck.jl", "DP")
        @test run_check("test/scaleTrack/singleDropletCheck.jl", "SP")
    end

    @testset "sync driver against the kernel" begin
        @test run_check("test/scaleTrack/syncDriverCheck.jl", "DP")
        @test run_check("test/scaleTrack/syncDriverCheck.jl", "SP")
    end

    @testset "Hilbert curve" begin
        @test run_check("test/scaleTrack/hilbertCheck.jl")
    end

    # ----------------------------------------------------------- regression
    # Against checksums, so a failure means the numbers moved -- which is
    # either a mistake or a deliberate change that has not been recorded yet.
    @testset "StokesFlow, thread count" begin
        # Trajectories depend only on the frozen velocity field, so the
        # particle state must be bitwise independent of the thread count.
        # The momentum source need not be: its per-cell accumulation order
        # is not.
        serial = joinpath(tmp, "checks_serial.txt")
        nt4 = joinpath(tmp, "checks_nt4.txt")
        script = "test/icoJuliaParcelFoam/cavity3D/testThreadedTracking.jl"

        @test run_check(script, serial; threads = "1,1")
        @test run_check(script, nt4; threads = "4,1")

        if isfile(serial) && isfile(nt4)
            a, b = read_checks(serial), read_checks(nt4)
            @test keys(a) == keys(b)
            for name in sort(collect(keys(a)))
                if name == "sumUTrans"
                    va = parse.(Float64, split(a[name]))
                    vb = parse.(Float64, split(b[name]))
                    scale = maximum(abs, va)
                    @test all(abs.(va .- vb) .<= 1e-6*max(scale, 1))
                else
                    @test a[name] == b[name]
                end
            end
        end
    end

    @testset "HumidAirDroplet, golden checksums" begin
        case = "run/buoyantHumidPimpleParcelFoam/freeFallCoolingEvaporation"
        golden = joinpath(ROOT, case, "checks_humid.txt")
        written = joinpath(tmp, "checks_humid.txt")

        @test run_check(joinpath(case, "testHumidTracking.jl"), written)
        if isfile(written)
            @test read_checks(written) == read_checks(golden)
        end
    end

end

end
