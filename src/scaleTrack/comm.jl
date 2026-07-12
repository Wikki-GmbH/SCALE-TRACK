#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Communication members and the MPI setup.  Ranks hosting a tracking device
# (GPU, or one rank per node for the CPU executor) become Masters that track
# particle chunks; the remaining ranks are Slaves that only serve their
# Eulerian partition and receive back source terms.

abstract type CommMember end

struct Master <: CommMember
    deviceNumber::label
    requiredEulerianRanks::Set{label}
    inquiringEulerianRanks::Vector{label}
end

Master(deviceNumber) = Master(deviceNumber, Set{label}(), Vector{label}())

struct Slave <: CommMember
    inquiringEulerianRanks::Vector{label}
end

Slave() = Slave(Vector{label}())

struct Comm{T<:CommMember}
    member::T
    communicator::MPI.Comm
    isMaster::Bool
    isHost::Bool
    rank::Integer
    jlRank::Integer
    masterRank::Integer
    size::Integer
end

function Comm(member, communicator)
    if MPI.Initialized()
        rank = MPI.Comm_rank(communicator)
        isMaster = (rank == 0)
        isHost = (typeof(member) == Master)
        return Comm(
            member,
            communicator,
            isMaster,
            isHost,
            rank,
            rank + 1,
            0,
            MPI.Comm_size(communicator)
        )
    else
        rank = 0
        return Comm(member, communicator, true, true, rank, rank + 1, 0, 1)
    end
end

function initComm(executor)
    if MPI.Initialized() && MPI.JULIA_TYPE_PTR_ATTR[]==0
        MPI.run_init_hooks()
    else
        # Construct communication with a single rank and executor.  The MPI
        # library may install its own SIGSEGV handler, which would break the
        # safepoint mechanism Julia's garbage collector uses to park the
        # threads — preserve Julia's handler across the initialization (the
        # solver guards its OpenFOAM signal setup the same way)
        SIGSEGV = 11 % Cint
        oldAction = Vector{UInt8}(undef, 512)
        ccall(
            :sigaction, Cint, (Cint, Ptr{Cvoid}, Ptr{UInt8}),
            SIGSEGV, C_NULL, oldAction
        )
        MPI.Init()
        ccall(
            :sigaction, Cint, (Cint, Ptr{UInt8}, Ptr{Cvoid}),
            SIGSEGV, oldAction, C_NULL
        )
    end

    nDevicesPerNode = count_devices_per_node(executor)
    shmComm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, 0)
    shmRank = MPI.Comm_rank(shmComm)
    nRanksPerNode = MPI.Allreduce(shmRank, max, shmComm) + 1
    hostStride = nRanksPerNode ÷ nDevicesPerNode
    # Prevent 0 stride
    hostStride = hostStride < 1 ? 1 : hostStride

    # Assign devices to ranks uniformly
    hostRanks =
        range(0, step=hostStride, length=min(nRanksPerNode, nDevicesPerNode))
    if shmRank in hostRanks
        deviceNumber = shmRank ÷ hostStride
        set_device!(deviceNumber, executor)
        comm = Comm(Master(deviceNumber), MPI.COMM_WORLD)
    else
        comm = Comm(Slave(), MPI.COMM_WORLD)
    end

    if !DebugComm && !comm.isMaster
        # Suppress output from non-master ranks
        redirect_stdout(devnull)
    end

    return comm
end
