#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Elementary types: numeric shorthands, small vectors, Eulerian and Lagrangian
# containers, and the executor tags

# A shorthand notation for custom types and multiplication operation for easier
# introduction of the type primitives into the code
struct LBL end
(*)(n, ::Type{LBL}) = label(n)
struct SCL end
(*)(n, ::Type{SCL}) = scalar(n)

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

# The Eulerian fields of one partition coupled in both directions: the carrier
# velocity U is read by the tracking, the momentum source UTrans is written
# back to the carrier phase
struct TwoWayEulerian{T}
    N::label
    U::T
    UTrans::T
end

function TwoWayEulerian{T}(N) where {T}
    TwoWayEulerian{T}(
        N,
        T(undef, N),
        T(undef, N)
    )
end
Adapt.@adapt_structure TwoWayEulerian

struct Time
    t::scalar
    Δt::scalar
end

struct BoundingBox{T}
    min::T
    max::T
end

function BoundingBox{T}() where {T}
    BoundingBox{T}(T(undef, 3), T(undef, 3))
end

Adapt.@adapt_structure BoundingBox

# A chunk of particles stored as a structure of arrays.
# Superscripts: c - carrier phase; d - dispersed phase
struct Chunk{T, A}
    N::label
    μᶜ::scalar
    ρ::scalar
    ρᵈByρᶜ::scalar
    boundingBox::BoundingBox{T}
    time::A
    d::T
    X::T
    Y::T
    Z::T
    U::T
    V::T
    W::T
end

function Chunk{T, A}(N, μᶜ, ρ, ρᵈByρᶜ) where {T, A}
    Chunk{T, A}(
        N,
        μᶜ,
        ρ,
        ρᵈByρᶜ,
        BoundingBox{T}(),
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

# Structs used for function tagging to identify on which backend the code is
# executed
struct CPU end
struct GPU end

# Helper to copy all struct data from host to device
function copy!(a, b)
    for n in fieldnames(typeof(a))
        if !(typeof(getfield(b, n)) <: Number)
            if (typeof(getfield(b, n)) <: AbstractArray)
                copyto!(getfield(a, n), getfield(b, n))
            else
                copy!(getfield(a, n), getfield(b, n))
            end
        end
    end
    return nothing
end
