#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Elementary types: numeric shorthands, small vectors, and the physics-model
# agnostic containers.  The Eulerian field-set types belong to the physics
# models.

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

const ScalarField = Vector{scalar}
Adapt.@adapt_structure ScalarField

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

# A chunk of particles stored as a structure of arrays.  Every model tracks
# position, velocity and diameter; model-specific per-particle arrays (e.g.
# the droplet temperature) live in the props named tuple, allocated by the
# model's parcel_props trait.  nSubSteps is the number of Lagrangian sub-steps
# per coupling time step.
struct Chunk{T, A, P}
    N::label
    nSubSteps::label
    boundingBox::BoundingBox{T}
    time::A
    d::T
    X::T
    Y::T
    Z::T
    U::T
    V::T
    W::T
    props::P
end

function Chunk{T, A}(N, nSubSteps, props::NamedTuple) where {T, A}
    Chunk{T, A, typeof(props)}(
        N,
        nSubSteps,
        BoundingBox{T}(),
        A(undef, 1),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        T(undef, N),
        props
    )
end

Adapt.@adapt_structure Chunk

# The per-particle working state the evolve skeleton advances:
# position and velocity are universal, the model keeps its extra evolving
# state (temperature, mass, ...) in the props named tuple.  Immutable — the
# skeleton and the model hooks advance it functionally.
struct ParcelState{P}
    pos::ScalarVec
    vel::ScalarVec
    props::P
end
