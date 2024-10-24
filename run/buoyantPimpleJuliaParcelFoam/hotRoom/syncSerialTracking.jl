#=
    SCALE-TRACK

    SCALE-TRACK is advancing two-way coupled Euler-Lagrange particle tracking
    to the realm of exascale computing enabling the simulation of dispersed
    multiphase flows at lower cost and energy consumption.  This is achieved
    through a coupling algorithm, which eliminates synchronisation barriers,
    and new cache-friendly data structures.

    Copyright (C) 2024 Sergey Lesnik
    Copyright (C) 2024 Henrik Rusche

    This file is part of SCALE-TRACK.

    SCALE-TRACK is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along
    with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.
=#

println("Load syncSerialTracking")

tNow = time()

using Random
# using Distributions  # Hangs when profiling with Nsight
using WriteVTK
using CUDA
using StaticArrays
using BenchmarkTools
import Adapt
import Base: *

###############################################################################
# Auxillary Methods

function timing(t, s)
    dt = round(time() - t, sigdigits=4)
    println(s, " in ", dt, " s")

    # Invoke flush to ensure immediate printing even when executed within C code
    flush(stdout)
    return time()
end

tNow = timing(tNow, "Loaded modules")

###############################################################################
# Type alias

const scalar = Float32
const label = Int32

# A shorthand notation for custom types and multiplication operation for easier
# introduction of the type primitives into the code
struct LBL end
(*)(n, ::Type{LBL}) = label(n)
struct SCL end
(*)(n, ::Type{SCL}) = scalar(n)


###############################################################################
# Structs and their constructors

struct LabelVec <: FieldVector{3, label}
    x::label
    y::label
    z::label
end

struct ScalarVec <: FieldVector{3, scalar}
    x::scalar
    y::scalar
    z::scalar
end

mutable struct MutScalarVec <: FieldVector{3, scalar}
    x::scalar
    y::scalar
    z::scalar
end

const VectorField = Vector{ScalarVec}
Adapt.@adapt_structure VectorField

# Mesh is limited to a cuboid defined by two points origin and ending with L
# containing length in each direction and N number of cells per direction
struct Mesh
    N::LabelVec
    origin::ScalarVec
    ending::ScalarVec
    L::ScalarVec
    Δ::ScalarVec
    rΔ::ScalarVec
    NyTimesNz::scalar
end

function Mesh(N::LabelVec, origin::ScalarVec, ending::ScalarVec)
    L = ending .- origin
    Δ = L ./ N
    rΔ = 1.0SCL ./ Δ
    Mesh(N, origin, ending, L, Δ, rΔ, N.y*N.z)
end

function Mesh(NInt::Integer, originR::Real, endingR::Real)
    N = LabelVec(NInt, NInt, NInt)
    origin = ScalarVec(originR, originR, originR)
    ending = ScalarVec(endingR, endingR, endingR)
    L = ending .- origin
    Δ = L ./ N
    rΔ = 1.0SCL ./ Δ
    Mesh(N, origin, ending, L, Δ, rΔ, N.y*N.z)
end

Adapt.@adapt_structure Mesh

struct Time
    t::scalar
    Δt::scalar
end

struct Chunk{T, A}
    N::label
    μ::scalar
    ρ::scalar
    time::A
    d::T
    X::T
    Y::T
    Z::T
    U::T
    V::T
    W::T
end

function Chunk{T, A}(N, μ, ρ) where {T, A}
    Chunk{T, A}(
        N,
        μ,
        ρ,
        A(undef, 1),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N)
    )
end

Adapt.@adapt_structure Chunk

# Structs used for function taging to identify on which backend the code is
# executed
struct CPU end
struct GPU end

###############################################################################
# Methods
function allocate_chunk(size, μ, ρ, ::CPU)
    return Chunk{Vector{scalar}, Vector{Time}}(size, μ, ρ)
end

function allocate_chunk(size, μ, ρ, ::GPU)
    return Chunk{CuVector{scalar}, CuVector{Time}}(size, μ, ρ)
end

function allocate_field(size, ::CPU)
    return VectorField(undef, size)
end

function allocate_field(size, ::GPU)
    return CuVector{ScalarVec}(undef, size)
end

function set_time!(chunk, t, Δt, ::CPU)
    chunk.time[1] = Time(t, Δt)
end

function set_time!(chunk, t, Δt, ::GPU)
    CUDA.@allowscalar chunk.time[1] = Time(t, Δt)
end

function increment_time!(chunk, Δt, ::CPU)
    set_time!(chunk, chunk.time[1].t + Δt, Δt, executor)
end

function increment_time!(chunk, Δt, ::GPU)
    CUDA.@allowscalar t = chunk.time[1].t
    set_time!(chunk, t + Δt, Δt, executor)
end

function init!(chunk, mesh, executor)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)

    rng = Random.default_rng()
    Random.seed!(rng, 19891)
    rand!(rng, c.d)
    rand!(rng, c.X)
    rand!(rng, c.Y)
    rand!(rng, c.Z)
    @. c.d = c.d*5e-3SCL + 5e-3SCL
    @. c.X = c.X*mesh.L.x + mesh.origin.x
    @. c.Y = c.Y*mesh.L.y + mesh.origin.y
    @. c.Z = c.Z*mesh.L.z + mesh.origin.z
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    return nothing
end

function init_random!(field)
    rng = Random.default_rng()
    Random.seed!(rng, 19891)
    rand!(rng, field)
    map!(x -> x*5e-3SCL .+ 5e-3SCL, field, field)
    return nothing
end

# Helper to copy all struct data from host to device
function copy!(a, b)
    for n in fieldnames(typeof(a))
        if !(typeof(getfield(b, n)) <: Number)
            copyto!(getfield(a, n), getfield(b, n))
        end
    end
    return nothing
end

# Find cell index given position
@inline function locate(x, y, z, mesh)
    # TODO unsafe_trunc instead of floor is 20% faster but then numbers within
    # -1 < n < 0 are truncated to 0 which leads to incorrect localization of
    # the particle
    i = floor(label, (x - mesh.origin.x)*mesh.rΔ.x)
    j = floor(label, (y - mesh.origin.y)*mesh.rΔ.y)
    k = floor(label, (z - mesh.origin.z)*mesh.rΔ.z)
    return (i, j, k, (abs(k)*mesh.NyTimesNz + abs(j)*mesh.N.x + abs(i) + 1LBL))
end

@inline function locate(pos::MutScalarVec, mesh)
    return locate(pos.x, pos.y, pos.z, mesh)
end

# Evolve particles using CPU
function evolve!(chunk, Uc, mesh, Δt, ::CPU)
    @inbounds begin
        increment_time!(chunk, Δt, CPU())
        nSteps = 10LBL
        ΔtP = chunk.time[1].Δt / nSteps
        for i = 1LBL:chunk.N
            evolve_particle!(chunk, Uc, i, ΔtP, mesh, nSteps)
        end
    end
    return nothing
end

# CUDA call with a setup that efficiently maps to the spesific GPU
function evolve!(chunk, Uc, mesh, Δt, ::GPU)
    if !haskey(reg, "deviceU")
        reg["deviceU"] = CuVector{ScalarVec}(undef, length(Uc))
    end
    copyto!(reg["deviceU"], Uc)

    increment_time!(chunk, Δt, GPU())

    kernel = @cuda launch=false evolve_on_device!(chunk, reg["deviceU"], mesh)
    config = launch_configuration(kernel.fun)
    threads = min(chunk.N, config.threads)
    blocks = cld(chunk.N, threads)

    CUDA.@sync kernel(chunk, reg["deviceU"], mesh; threads, blocks)
    return nothing
end

# Evolve particles using GPU
function evolve_on_device!(chunk, Uc, mesh)
    @inbounds begin
        nSteps = 10LBL
        Δtd = chunk.time[1].Δt / nSteps  # Time step for the dispersed phase
        nParticles = chunk.N
        i = (blockIdx().x - 1LBL) * blockDim().x + threadIdx().x
        if(i <= nParticles)
            evolve_particle!(chunk, Uc, i, Δtd, mesh, nSteps)
        end
    end
    return nothing
end

@inline function evolve_particle!(chunk, Uc, i, Δt, mesh, nSteps)
    @inbounds begin
        c = chunk
        μ = c.μ
        ρ = c.ρ
        ⌀ = c.d[i]
        pos = MutScalarVec(c.X[i], c.Y[i], c.Z[i])
        vel = MutScalarVec(c.U[i], c.V[i], c.W[i])
        posNew = MutScalarVec(c.X[i], c.Y[i], c.Z[i])
        velNew = MutScalarVec(c.U[i], c.V[i], c.W[i])

        I, J, K, posI::label = locate(pos, mesh)
        u = Uc[posI]

        for t = 1LBL:nSteps

            # Explicit Euler time integration with Stokes drag
            dragFactor = 18SCL*μ*Δt/(ρ*⌀^2)
            velNew .= vel .+ dragFactor.*(u .- vel)
            posNew .= pos .+ velNew.*Δt

            # Find the new position index and check whether the particle hits a
            # boundary.  If yes, then hold the corresponding position component
            # and mirror the velocity component.  In this way, the particle
            # will always stay within the domain.

            I, J, K, posNewI::label = locate(posNew, mesh)

            bi = (posNewI !== posI)
            if bi
                bx = (I < 0LBL) || (I >= mesh.N.x)
                by = (J < 0LBL) || (J >= mesh.N.y)
                bz = (K < 0LBL) || (K >= mesh.N.z)

                if bx
                    posNew.x = pos.x
                    velNew.x = -vel.x
                end

                if by
                    posNew.y = pos.y
                    velNew.y = -vel.y
                end

                if bz
                    posNew.z = pos.z
                    velNew.z = -vel.z
                end

                if (bx || by || bz)
                    I, J, K, posNewI = locate(posNew, mesh)
                end

                if (t !== nSteps && posNewI !== posI)
                    u = Uc[posNewI]
                    posI = posNewI
                end
            end
        end

        c.X[i] = posNew.x
        c.Y[i] = posNew.y
        c.Z[i] = posNew.z
        c.U[i] = velNew.x
        c.V[i] = velNew.y
        c.W[i] = velNew.z
    end
    return nothing
end

function write(chunk, ::CPU)
    c = chunk
    t = round(c.time[1].t, sigdigits=4)
    println("Writing time ", c.time[1].t, " as ", t)

    if !(haskey(reg, "paraview"))
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i, )) for i = 1:c.N]
        pvd = paraview_collection("particleTimeSeries")
        reg["paraview"] = Dict("cells" => cells, "pvd" => pvd)
    end

    pv = reg["paraview"]
    cells = pv["cells"]
    pvd = pv["pvd"]
    if !isdir("dataVTK")
        mkdir("dataVTK")
    end
    vtk_grid("dataVTK/particleFields_$(t).vtu", c.X, c.Y, c.Z, cells) do vtk
        vtk["U", VTKPointData()] = transpose(stack([c.U, c.V, c.W]))
        vtk["d", VTKPointData()] = c.d
        pvd[t] = vtk
    end
    return nothing
end

function write(chunk, ::GPU)
    c = chunk
    if !(haskey(reg, "hostChunk"))
        reg["hostChunk"] = Chunk{Vector{scalar}, Vector{Time}}(c.N, c.μ, c.ρ)
    end
    copy!(reg["hostChunk"], c)
    write(reg["hostChunk"], CPU())
    return nothing
end

function write_paraview_collection()
    if haskey(reg, "paraview")
        if haskey(reg["paraview"], "pvd")
            vtkCollection = reg["paraview"]["pvd"]
            println("Writing paraview collection to ", vtkCollection.path)
            flush(stdout)
            vtk_save(vtkCollection)
            return nothing
        end
    end
    println("No paraview collection written since no paraview instance found"
            *" in the registry")
    return nothing
end


###############################################################################
# Coupling to C++

reg = Dict()

function allocate_array_j(
        name::Cstring, size::Cint, nComponents::Cint, typeByteSize::Cint
    )::Ptr{Cdouble}
    GC.@preserve name
    s = unsafe_string(pointer(name))

    if Int(typeByteSize) !== Int(sizeof(scalar))
        throw(
            ErrorException(
                string(
                    "Size of Julia type (in Bytes): ", sizeof(scalar),
                    " is different from the extern type: ", typeByteSize
                )
            )
        )
    end

    # nComponents needs to be Int
    n = Int(nComponents)
    if (n == 1)
        reg[s] = Vector{scalar}(undef, size)
    else
        reg[s] = Vector{SVector{n, scalar}}(undef, size)
    end
    return Base.unsafe_convert(Ptr{Cdouble}, reg[s])
end

const allocate_array_ptr =
    @cfunction(allocate_array_j, Ptr{Cdouble}, (Cstring, Cint, Cint, Cint))

function evolve_cloud(Δt)
    println("Evolve particles")
    flush(stdout)

    tStart = time()

    evolve!(chunk, reg["U"], mesh, Δt, executor)

    tEvolve = time() - tStart
    if !firstPass
        global totalTime += tEvolve
    end
    global firstPass = false
    println("Lagrangian solver timings: current evolve = ",
        round(tEvolve, sigdigits=4), " s; total time = ",
        round(totalTime, sigdigits=4), " s"
    )
    return nothing
end

const evolve_cloud_ptr = @cfunction(evolve_cloud, Cvoid, (Cdouble,))

###############################################################################
# Global data init

# nParticles = 800_000_000  # RTX3090
# nParticles = 25_000_000  # RTX3090 fast
# nParticles = 1_000_000  # RTX3090 faster
# nParticles = 100_000  # GT710
nParticles = 100

# executor = GPU()
executor = CPU()

μ = 1e-3
ρ = 1.0
nCellsPerDirection = 20
origin = 0.0
ending = 1.0

#@show nParticles μ ρ nCellsPerDirection origin ending

mesh = Mesh(nCellsPerDirection, origin, ending)
chunk = allocate_chunk(nParticles, μ, ρ, executor)
tNow = timing(tNow, "Allocated particle chunk")
init!(chunk, mesh, executor)
tNow = timing(tNow, "Initialized particle chunk")

# Write initial state
write(chunk, executor)
tNow = timing(tNow, "Written VTK data")

# Create random velocity field when running in REPL
#if isinteractive()
#    reg["U"] = allocate_field(nCellsPerDirection^3, CPU())
#    tNow = timing(tNow, "Allocated velocity field")
#    init_random!(reg["U"])
#    tNow = timing(tNow, "Initialized velocity field")
#end

totalTime = 0.0
firstPass = true

println("Load syncSerialTracking done\n")
flush(stdout)
flush(stderr)

###############################################################################
