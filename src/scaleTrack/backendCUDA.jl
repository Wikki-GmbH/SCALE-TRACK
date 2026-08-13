#=
    SPDX-License-Identifier: GPL-3.0-or-later

    This file is part of SCALE-TRACK, distributed under the GNU GPL v3 or
    later.  See scaleTrack.jl or <http://www.gnu.org/licenses/> for details.

    Copyright (C) 2024-2026 Sergey Lesnik
    Copyright (C) 2024-2026 Henrik Rusche
=#

# CUDA backend: the vendor primitives the shared GPU executor dispatches to.

GPU() = CUDAGPU()

device_count(::CUDAGPU) = length(CUDA.devices())
set_device!(deviceNumber, ::CUDAGPU) = CUDA.device!(deviceNumber)
current_device(::CUDAGPU) = CUDA.device()
device_rng(::CUDAGPU) = CUDA.default_rng()
synchronize_device(::CUDAGPU) = CUDA.synchronize()

device_vector_type(::CUDAGPU, ::Type{T}) where {T} = CuVector{T}
device_ptr_vector_type(::CUDAGPU, ::Type{T}) where {T} = CuDeviceVector{T, 1}
device_convert(x, ::CUDAGPU) = cudaconvert(x)

# Host access to a single element of a device array
scalar_get(arr, i, ::CUDAGPU) = CUDA.@allowscalar arr[i]
scalar_set!(arr, i, v, ::CUDAGPU) = CUDA.@allowscalar arr[i] = v

# In-kernel index intrinsics and block-wide barrier
@inline thread_index(::CUDAGPU) = threadIdx().x
@inline block_index(::CUDAGPU) = blockIdx().x
@inline block_dim(::CUDAGPU) = blockDim().x
@inline sync_block(::CUDAGPU) = sync_threads()

# Static shared memory.  The length is passed as a Val so that it stays a
# compile-time constant, which the allocation requires.
@inline static_shared(::Type{T}, ::Val{N}, ::CUDAGPU) where {T, N} =
    CUDA.CuStaticSharedArray(T, N)

# Atomically accumulate into the source fields.  A vector field is addressed
# per component: ScalarVec is immutable and cannot be mutated atomically the
# straightforward way, so the field is reinterpreted as scalar and the
# component reached through a pointer.
@inline function atomic_add_component!(field, cellI, comp, val, ::CUDAGPU)
    ptr = pointer(reinterpret(scalar, field), (cellI-1LBL)*3LBL + comp)
    CUDA.atomic_add!(ptr, val)
    return nothing
end

@inline function atomic_add!(field, cellI, val, ::CUDAGPU)
    CUDA.atomic_add!(pointer(field, cellI), val)
    return nothing
end

# Compare-and-swap returning the value found and whether it was replaced
@inline function atomic_replace!(arr, i, expected, desired, ::CUDAGPU)
    old = CUDA.atomic_cas!(pointer(arr, i), expected, desired)
    return old, old === expected
end

# CUDA offers no atomic minimum or maximum for floating point, so the
# comparison runs on the bit pattern: over the non-negative floats the signed
# integer order agrees with the float order, and over the negative ones the
# unsigned order reverses it.  The sign of the candidate therefore selects
# both the integer type and the operation.
@inline ordered_signed(x::Float32) = reinterpret(Int32, x)
@inline ordered_signed(x::Float64) = reinterpret(Int64, x)
@inline ordered_unsigned(x::Float32) = reinterpret(UInt32, x)
@inline ordered_unsigned(x::Float64) = reinterpret(UInt64, x)

@inline ordered_signed_ptr(ptr::Core.LLVMPtr{Float32, A}) where {A} =
    reinterpret(Core.LLVMPtr{Int32, A}, ptr)
@inline ordered_signed_ptr(ptr::Core.LLVMPtr{Float64, A}) where {A} =
    reinterpret(Core.LLVMPtr{Int64, A}, ptr)
@inline ordered_unsigned_ptr(ptr::Core.LLVMPtr{Float32, A}) where {A} =
    reinterpret(Core.LLVMPtr{UInt32, A}, ptr)
@inline ordered_unsigned_ptr(ptr::Core.LLVMPtr{Float64, A}) where {A} =
    reinterpret(Core.LLVMPtr{UInt64, A}, ptr)

@inline function atomic_min!(arr, i, val, ::CUDAGPU)
    ptr = pointer(arr, i)
    if val >= zero(val)
        CUDA.atomic_min!(ordered_signed_ptr(ptr), ordered_signed(val))
    else
        CUDA.atomic_max!(ordered_unsigned_ptr(ptr), ordered_unsigned(val))
    end
    return nothing
end

@inline function atomic_max!(arr, i, val, ::CUDAGPU)
    ptr = pointer(arr, i)
    if val >= zero(val)
        CUDA.atomic_max!(ordered_signed_ptr(ptr), ordered_signed(val))
    else
        CUDA.atomic_min!(ordered_unsigned_ptr(ptr), ordered_unsigned(val))
    end
    return nothing
end

compile_kernel(f, args, ::CUDAGPU) = @cuda launch=false f(args...)

function kernel_config(kernel, ::CUDAGPU)
    config = launch_configuration(kernel.fun)
    return (threads = config.threads, blocks = config.blocks)
end

launch_kernel!(kernel, args, threads, blocks, ::CUDAGPU) =
    kernel(args...; threads, blocks)
