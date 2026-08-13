#=
    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2026 Sergey Lesnik
=#

# ROCm (AMD) backend: the vendor primitives the shared GPU executor
# dispatches to.
#
# AMD terminology maps onto the CUDA one used throughout the library as
# workitem -> thread, workgroup -> block, so "threads" and "blocks" below
# denote the workgroup size and the number of workgroups.

const Atomix = AMDGPU.Atomix
const UnsafeAtomics = AMDGPU.Device.UnsafeAtomics

# Device pointer to a scalar in global memory, used to reach a single
# component of a vector field
const ScalarPtr = Core.LLVMPtr{scalar, AMDGPU.Device.AS.Global}

GPU() = ROCmGPU()

device_count(::ROCmGPU) = length(AMDGPU.devices())
current_device(::ROCmGPU) = AMDGPU.device()
device_rng(::ROCmGPU) = AMDGPU.rocrand_rng()
synchronize_device(::ROCmGPU) = AMDGPU.synchronize()

# The device number is counted from zero, as on the CUDA side, while the AMDGPU
# device list is one based
set_device!(deviceNumber, ::ROCmGPU) =
    AMDGPU.device!(AMDGPU.devices()[deviceNumber + 1])

device_vector_type(::ROCmGPU, ::Type{T}) where {T} = ROCArray{T, 1}
device_ptr_vector_type(::ROCmGPU, ::Type{T}) where {T} =
    AMDGPU.Device.ROCDeviceVector{T, AMDGPU.Device.AS.Global}
device_convert(x, ::ROCmGPU) = AMDGPU.rocconvert(x)

# Host access to a single element of a device array
scalar_get(arr, i, ::ROCmGPU) = AMDGPU.@allowscalar arr[i]
scalar_set!(arr, i, v, ::ROCmGPU) = AMDGPU.@allowscalar arr[i] = v

# In-kernel index intrinsics and block-wide barrier
@inline thread_index(::ROCmGPU) = AMDGPU.workitemIdx().x
@inline block_index(::ROCmGPU) = AMDGPU.workgroupIdx().x
@inline block_dim(::ROCmGPU) = AMDGPU.workgroupDim().x
@inline sync_block(::ROCmGPU) = AMDGPU.sync_workgroup()

# Static shared memory (LDS).  Unlike the CUDA equivalent this is a macro, so
# the length has to reach it as a literal -- hence the Val.
@inline static_shared(::Type{T}, ::Val{N}, ::ROCmGPU) where {T, N} =
    AMDGPU.@ROCStaticLocalArray(T, N)

# Atomically accumulate into the source fields.  A vector field is addressed
# per component: ScalarVec is immutable and cannot be mutated atomically the
# straightforward way, so the component is reached through the pointer of its
# cell.  Reinterpreting the field itself is not an option here -- on a device
# array that falls back to the generic ReinterpretArray, whose padding check
# allocates and cannot be compiled for the device.
@inline function atomic_add_component!(field, cellI, comp, val, ::ROCmGPU)
    ptr = reinterpret(ScalarPtr, pointer(field, cellI))
    UnsafeAtomics.modify!(
        ptr + (comp - 1LBL)*sizeof(scalar), +, val, UnsafeAtomics.monotonic
    )
    return nothing
end

@inline function atomic_add!(field, cellI, val, ::ROCmGPU)
    UnsafeAtomics.modify!(
        pointer(field, cellI), +, val, UnsafeAtomics.monotonic
    )
    return nothing
end

# Compare-and-swap returning the value found and whether it was replaced
@inline function atomic_replace!(arr, i, expected, desired, ::ROCmGPU)
    old, success = Atomix.replace!(
        Atomix.IndexableRef(arr, (i,)), expected, desired,
        Atomix.seq_cst, Atomix.seq_cst
    )
    return old, success
end

# ROCm has atomic min/max, so no compare-and-swap loop is needed here.
# Addressing through the pointer keeps this working on shared memory too,
# where the address space differs.
@inline function atomic_min!(arr, i, val, ::ROCmGPU)
    UnsafeAtomics.modify!(pointer(arr, i), min, val, UnsafeAtomics.monotonic)
    return nothing
end

@inline function atomic_max!(arr, i, val, ::ROCmGPU)
    UnsafeAtomics.modify!(pointer(arr, i), max, val, UnsafeAtomics.monotonic)
    return nothing
end

compile_kernel(f, args, ::ROCmGPU) = AMDGPU.@roc launch=false f(args...)

function kernel_config(kernel, ::ROCmGPU)
    config = AMDGPU.launch_configuration(kernel)
    return (threads = config.groupsize, blocks = config.gridsize)
end

launch_kernel!(kernel, args, threads, blocks, ::ROCmGPU) =
    kernel(args...; groupsize = threads, gridsize = blocks)
