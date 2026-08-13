#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

#=
    Model-independent particle evolution skeleton.  The physical model (the
    forces, transfer laws, carrier fields and source accumulation) is provided
    by the physics model; the skeleton owns
    everything every model shares: the mesh localization, the sub-stepping,
    the boundary bounce, the flush of the sources on a cell change and the
    parcel state write-back.
=#

# Initial values of a model's per-parcel property arrays.  The drivers apply
# this to every chunk before the initializer runs, so an initializer only has
# to set what its case wants different.  Zero is a usable state for a property
# that merely accumulates; one that enters a law as a divisor or an argument
# of a nonlinear function has to name a value here, or a cloud that keeps the
# default state cannot be evolved at all.
init_props!(chunk, model) = foreach(a -> fill!(a, 0.0), values(chunk.props))

# Default particle initialization: diameters and positions uniformly random
# over their ranges, velocities zero, model properties as the model defines
# them.  Cases with different initial conditions pass their own function to
# init_async_tracking! / init_sync_tracking!.
#
# The driver hands every initializer the chunk's index within the whole cloud
# and the number of chunks the cloud is split into, which is what lets an
# initializer give each chunk its own region.
# Here the index only seeds the random number generator, so that two chunks
# do not receive the same particles.
function init!(chunk, mesh, executor, iChunk = 19891, nChunksGlobal = 1)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)

    rng = default_rng(executor)
    Random.seed!(rng, iChunk)
    fill!(c.boundingBox.min, 0.0)
    fill!(c.boundingBox.max, 0.0)
    rand!(rng, c.d)
    rand!(rng, c.X)
    rand!(rng, c.Y)
    rand!(rng, c.Z)
    @. c.d = c.d*5e-3SCL + 5e-3SCL
    @. c.X = c.X*mesh.L.x + mesh.origin.x
    @. c.Y = c.Y*mesh.L.y + mesh.origin.y
    @. c.Z = c.Z*mesh.L.z + mesh.origin.z
    hold_inside!(c, mesh)
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    return nothing
end

#=
    Hold every particle just inside the domain.

    A coordinate landing exactly on the far boundary is localized into a
    partition one past the last, which indexes past the end of the Eulerian
    containers -- on a device that is an out-of-bounds access, and on the
    host it surfaces later as a bounding box that covers a partition that
    does not exist.

    Two ways in.  The device generators draw from an interval that is
    half-open at best, and closed at the top on ROCm, so scaling a draw to
    the domain can produce the boundary outright.  And where a cell index is
    added to an offset within the cell, as the space-filling layout does, the
    sum costs the offset its low bits and an offset just short of the next
    cell rounds into it.

    Rare per draw either way -- which is what makes it worth doing here
    rather than leaving to chance: a cloud is initialized once, but it is
    initialized with millions of draws, and a run that grows the cloud or
    splits it over more chunks takes more of them.
=#
function hold_inside!(chunk, mesh)
    c = chunk
    xMax = prevfloat(mesh.ending.x)
    yMax = prevfloat(mesh.ending.y)
    zMax = prevfloat(mesh.ending.z)
    @. c.X = min(c.X, xMax)
    @. c.Y = min(c.Y, yMax)
    @. c.Z = min(c.Z, zMax)
    return nothing
end

#=
    Hilbert-curve particle positions.

    A chunk is the unit the tracking moves and evolves, so its particles
    should sit close together in space and, within it, particles close in
    memory should be close in space too: both keep the carrier-field accesses
    of a kernel inside a small region.  Placing the cloud along a
    space-filling curve gives that.  Each chunk takes a contiguous stretch of
    the curve, one stretch per chunk over the whole cloud, and every particle
    picks a random point of its own stretch.

    The curve walks a cube of its own, of a resolution set independently of
    the mesh.  A point of that cube maps to a cell by scaling, and the
    particle is placed uniformly at random within the cell.  nBits therefore
    only needs to resolve the mesh -- beyond that the extra points fall in
    the same cells.
=#

# The stretch of the curve a chunk occupies: the iChunk-th of nChunksGlobal
# equal parts.  Splitting the curve rather than the domain is what makes the
# chunks of all tracking masters together cover the cloud exactly once.
function chunk_curve_range(iChunk, nChunksGlobal, nBits)
    lCurve = curve_length(nBits)
    return (
        ((iChunk - 1)*lCurve) ÷ nChunksGlobal,
        (iChunk*lCurve) ÷ nChunksGlobal - 1
    )
end

# Place a chunk's particles on its stretch of the curve.  Positions only: the
# caller sets whatever else its model needs.
function init_hilbert_positions!(
    chunk, mesh, executor, iChunk, nChunksGlobal; nBits = 10, randSeed = iChunk
)
    c = chunk
    hStart, hEnd = chunk_curve_range(iChunk, nChunksGlobal, nBits)

    # The offset within the cell, drawn on whichever device holds the chunk
    deviceRng = default_rng(executor)
    Random.seed!(deviceRng, randSeed)
    rand!(deviceRng, c.X)
    rand!(deviceRng, c.Y)
    rand!(deviceRng, c.Z)

    # The curve is walked on the host: the transform is inherently serial in
    # its bit planes and is run once per particle at initialization only
    cellX = Vector{scalar}(undef, c.N)
    cellY = Vector{scalar}(undef, c.N)
    cellZ = Vector{scalar}(undef, c.N)
    hostRng = Random.MersenneTwister(randSeed)
    stretch = hStart:hEnd
    for j in 1:c.N
        px, py, pz = hilbert_point(rand(hostRng, stretch), nBits)
        cellX[j] = (mesh.N.x*px) >> nBits
        cellY[j] = (mesh.N.y*py) >> nBits
        cellZ[j] = (mesh.N.z*pz) >> nBits
    end

    # similar() keeps this working for either executor: a device array for a
    # chunk on a device, a host array for one in host memory
    cells = similar(c.X)
    copyto!(cells, cellX)
    @. c.X = mesh.origin.x + mesh.Δ.x*(cells + c.X)
    copyto!(cells, cellY)
    @. c.Y = mesh.origin.y + mesh.Δ.y*(cells + c.Y)
    copyto!(cells, cellZ)
    @. c.Z = mesh.origin.z + mesh.Δ.z*(cells + c.Z)
    hold_inside!(c, mesh)

    return nothing
end

# Fill a host field with uniformly random values from a fixed seed
function init_random!(field, interval, offset)
    rng = Random.default_rng()
    Random.seed!(rng, 19891)
    rand!(rng, field)
    map!(x -> x*interval .+ offset, field, field)
    return nothing
end

# Boundary bounce: hold the position component and mirror the velocity
# component of the proposed state, based on the state before the sub-step
@inline function hold_and_mirror(prop::ParcelState, old::ParcelState, k)
    @inbounds ParcelState(
        Base.setindex(prop.pos, old.pos[k], k),
        Base.setindex(prop.vel, -old.vel[k], k),
        prop.props
    )
end

# Integrate a single particle over nSteps sub-steps of size Δt.  eulerianArr
# holds one Eulerian container per Lagrangian partition; the particle reads
# the carrier state from and accumulates its sources into the partition it
# currently resides in.
@inline function evolve_particle!(
    chunk, model, eulerianArr, i, Δt, mesh, nSteps, executor
)
    @inbounds begin
        c = chunk
        parcel = load_parcel(model, c, i)

        I, J, K, posI::label, partitionI::label = locate(parcel.pos, mesh)
        eulerian = eulerianArr[partitionI]
        carrier = load_carrier(model, eulerian, posI)
        acc = init_sources(model, eulerian, parcel)

        for t = 1LBL:nSteps

            # Model physics: propose the state after one sub-step and
            # accumulate the source contributions
            prop, acc = substep(model, parcel, carrier, acc, Δt)

            # Find the new position index and check whether the parcel hits a
            # boundary.  If yes, then hold the corresponding position component
            # and mirror the velocity component.  In this way, the parcel
            # will always stay within the domain.

            I, J, K, posNewI::label, partitionNewI::label =
                locate(prop.pos, mesh)

            # Has the parcel moved to another cell or partition?
            if (posNewI != posI) || (partitionNewI != partitionI)

                # Has the parcel hit a boundary?
                bx = (I < 0LBL) || (I >= mesh.N.x)
                by = (J < 0LBL) || (J >= mesh.N.y)
                bz = (K < 0LBL) || (K >= mesh.N.z)

                if bx
                    prop = hold_and_mirror(prop, parcel, 1)
                    acc = bounce_source(model, acc, parcel.vel, 1)
                end

                if by
                    prop = hold_and_mirror(prop, parcel, 2)
                    acc = bounce_source(model, acc, parcel.vel, 2)
                end

                if bz
                    prop = hold_and_mirror(prop, parcel, 3)
                    acc = bounce_source(model, acc, parcel.vel, 3)
                end

                # Reevaluate the cell and partition index the parcel is in,
                # since it may have changed after the boundary hit.
                if (bx || by || bz)
                    I, J, K, posNewI, partitionNewI = locate(prop.pos, mesh)
                end

                if (partitionNewI != partitionI)
                    # Reset pointer to eulerian
                    eulerian = eulerianArr[partitionI]
                end

                if (t != nSteps && posNewI != posI)
                    acc = flush_sources!(
                        model, eulerian, acc, prop, posI, executor
                    )
                    carrier = reload_carrier(model, eulerian, posNewI)
                    posI = posNewI
                end
            end

            parcel = prop
        end

        flush_sources!(model, eulerian, acc, parcel, posI, executor)
        store_parcel!(model, c, i, parcel)
        update!(c.boundingBox, parcel.pos, executor)
    end
    return nothing
end
