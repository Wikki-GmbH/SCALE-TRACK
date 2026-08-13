#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Structured mesh description and particle localization.  The mesh mirrors the
# case's mesh description on the Julia side and carries the Lagrangian domain
# decomposition, which is independent of the Eulerian one.

# Mesh is limited to a cuboid defined by two points origin and ending with L
# containing length in each direction and N number of cells per direction
struct Mesh
    N::LabelVec
    origin::ScalarVec
    ending::ScalarVec
    L::ScalarVec
    Δ::ScalarVec
    rΔ::ScalarVec
    decomposition::LabelVec
    decompositionXTimesY::label
    partitionN::LabelVec
    partitionNxTimesNy::label
end

function construct_mesh(N, origin, ending, decomposition)
    L = ending .- origin
    Δ = L ./ N
    rΔ = 1.0SCL ./ Δ
    if (sum(rem.(N, decomposition)) != 0)
        throw(
            ErrorException(
                string(
                    "Each partition must have the same number of cells per",
                    " direction.  The current setup does not comply:\n",
                    " N = ", N, " decomposition = ", decomposition
                )
            )
        )
    end
    decompositionXTimesY = decomposition[1]*decomposition[2]
    partitionN = N ./ decomposition
    partitionNxTimesNy = partitionN[1]*partitionN[2]
    Mesh(
        N, origin, ending, L, Δ, rΔ, decomposition, decompositionXTimesY,
        partitionN, partitionNxTimesNy
    )
end

function Mesh(
    NInt::Integer, originR::Real, endingR::Real, decomposition = (1, 1, 1)
)
    N = [NInt, NInt, NInt]
    origin = [originR, originR, originR]
    ending = [endingR, endingR, endingR]
    construct_mesh(N, origin, ending, decomposition)
end

Adapt.@adapt_structure Mesh

# Round towards -Inf, then convert to the label type without an exception
# path: raising InexactError boxes the value, which on a device means an
# allocation and the host-side service task that comes with it.  Rounding has
# to come first -- truncation towards zero would put -1 < v < 0 in cell 0
# instead of -1.  Out-of-range values and NaN are not reported; on a device
# they could not be reported usefully.
@inline floor_label(v) = unsafe_trunc(label, floor(v))

# Compute cell and partition indices along each direction.  Indices start at 0.
@inline function locate_ijk(x, y, z, mesh)
    iGlobal = floor_label((x - mesh.origin.x)*mesh.rΔ.x)
    jGlobal = floor_label((y - mesh.origin.y)*mesh.rΔ.y)
    kGlobal = floor_label((z - mesh.origin.z)*mesh.rΔ.z)
    iPartition, iLocal = divrem(iGlobal, mesh.partitionN.x)
    jPartition, jLocal = divrem(jGlobal, mesh.partitionN.y)
    kPartition, kLocal = divrem(kGlobal, mesh.partitionN.z)
    return (iGlobal, jGlobal, kGlobal, iLocal, jLocal, kLocal, iPartition,
        jPartition, kPartition
    )
end

# Compute linear cell index, the corresponding indices along each direction and
# linear partition index.  Linear indices start at 1 and direction indices at 0.
@inline function locate(x, y, z, mesh)
    iGlobal, jGlobal, kGlobal, iLocal, jLocal, kLocal, iPartition, jPartition,
    kPartition = locate_ijk(x, y, z, mesh)
    partitionI = (
        abs(kPartition)*mesh.decompositionXTimesY
        + abs(jPartition)*mesh.decomposition.x + abs(iPartition) + 1LBL
    )
    posI = (
        abs(kLocal)*mesh.partitionNxTimesNy
        + abs(jLocal)*mesh.partitionN.x + abs(iLocal) + 1LBL
    )
    return (iGlobal, jGlobal, kGlobal, posI, partitionI)
end

@inline function locate(pos, mesh)
    return locate(pos.x, pos.y, pos.z, mesh)
end
