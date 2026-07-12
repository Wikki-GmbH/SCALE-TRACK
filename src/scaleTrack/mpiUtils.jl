#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# Allocation-free wrappers around MPI calls.  MPI.jl's plain variants allocate
# (e.g. the flag or the status) on every call; the Julia GC is triggered only
# sporadically during a run, so use these wrappers in communication loops.

# Wrapper for MPI non-blocking synchronous send unavailable in MPI.jl
function comm_Issend(buf::MPI.Buffer, dest::Integer, tag::Integer, comm::MPI.Comm,
        req::MPI.AbstractRequest=MPI.Request()
)
    @assert MPI.isnull(req)
    # int MPI_Issend(const void* buf, int count, MPI_Datatype datatype, int
    #               dest, int tag, MPI_Comm comm, MPI_Request *request)
    MPI.API.MPI_Issend(buf.data, buf.count, buf.datatype, dest, tag, comm, req)
    MPI.setbuffer!(req, buf)
    return req
end

# Wrapper for MPI Iprobe that does not allocate flag
function comm_Iprobe(
    comm::MPI.Comm, flag, status=nothing;
    source::Integer=MPI.API.MPI_ANY_SOURCE[], tag::Integer=MPI.API.MPI_ANY_TAG[]
)
    MPI.API.MPI_Iprobe(
        source, tag, comm, flag, something(status, MPI.API.MPI_STATUS_IGNORE[])
    )
    return flag[] != 0
end

# Wrapper for MPI Test that does not allocate flag
function comm_test(
    req::MPI.AbstractRequest, flag,
    status::Union{Ref{MPI.Status}, Nothing}=nothing
)
    # int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
    MPI.API.MPI_Test(req, flag, something(status, MPI.API.MPI_STATUS_IGNORE[]))
    if MPI.isnull(req)
        MPI.setbuffer!(req, nothing)
    end
    return flag[] != 0
end

# Alternative implementation of Base.wait that allocates flag only once
function comm_wait(req::MPI.Request)
    flag = Ref{Cint}()
    while !comm_test(req, flag)
        yield()
    end
end
