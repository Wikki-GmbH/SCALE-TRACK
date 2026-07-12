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

# Default particle initialization: diameters and positions uniformly random
# over their ranges, velocities zero, model-specific parcel arrays zero.
# Cases with different initial conditions pass their own function to
# init_async_tracking! / init_sync_tracking!.
function init!(chunk, mesh, executor, randSeed=19891)
    c = chunk
    set_time!(c, 0.0, 0.0, executor)

    rng = default_rng(executor)
    Random.seed!(rng, randSeed)
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
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
    foreach(a -> fill!(a, 0.0), values(c.props))
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
