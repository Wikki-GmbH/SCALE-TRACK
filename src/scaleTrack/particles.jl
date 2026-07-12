#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Executor-independent particle physics: initialization, the per-particle
# time integration and the parcel state used for the two-way coupling.  The
# executor-specific accumulation of the sources (update!) lives in
# executorCPU.jl and executorGPU.jl.

# Number of Lagrangian sub-steps per coupling time step
const nTrackingSubSteps = 10LBL

# Default particle initialization: diameters and positions uniformly random
# over their ranges, velocities zero.  Cases with different initial conditions
# pass their own function to init_async_tracking! / init_sync_tracking!.
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
    # Clustered init in a part of the domain
    # @. c.X = c.X*mesh.Δ.x*mesh.partitionN.x*0.5*randSeed + mesh.origin.x + 0.6*mesh.L.x
    # @. c.Y = c.Y*mesh.Δ.y*mesh.partitionN.y*0.5*randSeed + mesh.origin.y + 0.6*mesh.L.y
    # @. c.Z = c.Z*mesh.Δ.z*mesh.partitionN.z*0.5*randSeed + mesh.origin.z + 0.6*mesh.L.z
    fill!(c.U, 0.0)
    fill!(c.V, 0.0)
    fill!(c.W, 0.0)
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

@inline function set_parcel_state(::TwoWayEulerian, velocity)
    return ScalarVec(velocity)
end

# Apply the change in velocity to the source due to bounce at boundary
@inline function update_at_boundary!(
    state0, ::TwoWayEulerian, velocity, componentI
)
    @inbounds @reset state0[componentI] -= 2SCL*velocity[componentI]
    return state0
end

# Integrate a single particle over nSteps sub-steps of size Δt.  eulerianArr
# holds one TwoWayEulerian per Lagrangian partition; the particle reads the
# carrier velocity from and accumulates its momentum source into the partition
# it currently resides in.
@inline function evolve_particle!(
    chunk, eulerianArr, i, Δt, mesh, nSteps, executor
)
    @inbounds begin
        c = chunk
        μᶜ = c.μᶜ
        ρᵈ = c.ρ
        ρᵈByρᶜ = c.ρᵈByρᶜ
        ⌀ = c.d[i]
        mᵈByρᶜ = ρᵈByρᶜ*π*⌀^3/6SCL
        pos = MutScalarVec(c.X[i], c.Y[i], c.Z[i])
        vel = MutScalarVec(c.U[i], c.V[i], c.W[i])
        posNew = MutScalarVec(c.X[i], c.Y[i], c.Z[i])
        velNew = MutScalarVec(c.U[i], c.V[i], c.W[i])

        I, J, K, posI::label, partitionI::label = locate(pos, mesh)
        eulerian = eulerianArr[partitionI]
        uᶜ = eulerian.U[posI]
        state0 = set_parcel_state(eulerian, vel)

        for t = 1LBL:nSteps

            # Implicit Euler time integration with Stokes drag
            dragFactor = 18SCL*μᶜ*Δt/(ρᵈ*⌀^2)
            velNew .= (vel .+ dragFactor.*uᶜ)./(1 + dragFactor)
            posNew .= pos .+ velNew.*Δt

            # Find the new position index and check whether the parcel hits a
            # boundary.  If yes, then hold the corresponding position component
            # and mirror the velocity component.  In this way, the parcel
            # will always stay within the domain.

            I, J, K, posNewI::label, partitionNewI::label = locate(posNew, mesh)

            # Has the parcel moved to another cell or partition?
            if (posNewI != posI) || (partitionNewI != partitionI)

                # Has the parcel hit a boundary?
                bx = (I < 0LBL) || (I >= mesh.N.x)
                by = (J < 0LBL) || (J >= mesh.N.y)
                bz = (K < 0LBL) || (K >= mesh.N.z)

                if bx
                    posNew.x = pos.x
                    velNew.x = -vel.x
                    state0 = update_at_boundary!(state0, eulerian, vel, 1)
                end

                if by
                    posNew.y = pos.y
                    velNew.y = -vel.y
                    state0 = update_at_boundary!(state0, eulerian, vel, 2)
                end

                if bz
                    posNew.z = pos.z
                    velNew.z = -vel.z
                    state0 = update_at_boundary!(state0, eulerian, vel, 3)
                end

                # Reevaluate the cell and partition index the parcel is in,
                # since it may have changed after the boundary hit.
                if (bx || by || bz)
                    I, J, K, posNewI, partitionNewI = locate(posNew, mesh)
                end

                if (partitionNewI != partitionI)
                    # Reset pointer to eulerian
                    eulerian = eulerianArr[partitionI]
                end

                if (t != nSteps && posNewI != posI)
                    state0 = update!(
                        eulerian, state0, velNew, posI, mᵈByρᶜ, executor
                    )
                    uᶜ = eulerian.U[posNewI]
                    posI = posNewI
                end
            end

            vel .= velNew
            pos .= posNew
        end

        c.X[i] = posNew.x
        c.Y[i] = posNew.y
        c.Z[i] = posNew.z
        c.U[i] = velNew.x
        c.V[i] = velNew.y
        c.W[i] = velNew.z
        update!(eulerian, state0, velNew, posI, mᵈByρᶜ, executor)
        update!(c.boundingBox, posNew, executor)
    end
    return nothing
end
