#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# VTK output of the particle state into the case's dataVTK directory and the
# accompanying ParaView collection (particleTimeSeries.pvd)

function write(chunk, ::CPU)
    c = chunk
    t = round(c.time[1].t, sigdigits = 4)
    println("Writing time ", c.time[1].t, " as ", t)

    if !(haskey(reg, "paraview"))
        cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i = 1:c.N]
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
        # Model-specific per-particle fields (e.g. the droplet temperature)
        for (name, arr) in pairs(c.props)
            vtk[string(name), VTKPointData()] = arr
        end
        pvd[t] = vtk
    end
    return nothing
end

# A host-memory chunk with the same layout as the given (device) chunk
function allocate_host_chunk(c::Chunk)
    T = Vector{scalar}
    props = NamedTuple{keys(c.props)}(
        map(_ -> T(undef, c.N), values(c.props))
    )
    return Chunk{T, Vector{Time}}(c.N, c.nSubSteps, props)
end

function write(chunk, ::GPU)
    c = chunk
    if !(haskey(reg, "hostChunk"))
        reg["hostChunk"] = allocate_host_chunk(c)
    end
    copy!(reg["hostChunk"], c)
    write(reg["hostChunk"], CPU())
    return nothing
end

# Write the chunk data from the tracking master.  Hold chunkTransfers so a
# concurrent evolve does not mutate the particle state mid-write (the CPU
# writes in place, the GPU copies device to host first).
function write(chunks, comm::Comm{Master}, executor)
    # TODO Enable writing for all chunks, not only the first one
    if comm.isMaster
        lock(control.locks.chunkTransfers) do
            write(first(chunks), executor)
        end
    end
end

function write(chunk, ::Comm{<:CommMember}, executor) end

function write_paraview_collection(::Comm{Master})
    write_paraview_collection()
end

function write_paraview_collection(::Comm{<:CommMember}) end

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
    println(
        "No paraview collection written since no paraview instance found"
        *
        " in the registry"
    )
    return nothing
end
