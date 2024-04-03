// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// This is a copy of FP6-LLM kernel code: https://arxiv.org/abs/2401.14112

#ifndef PTX_CP_ASYNC_CUH
#define PTX_CP_ASYNC_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <int SizeInBytes>
__device__ __forceinline__ void cp_async(half* smem_ptr,
                                         const half* global_ptr,
                                         bool pred_guard = true)
{
    static_assert(SizeInBytes == 16, "Size is not supported");
    unsigned smem_int_ptr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{ \n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)pred_guard),
        "r"(smem_int_ptr),
        "l"(global_ptr),
        "n"(SizeInBytes));
}

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
__device__ __forceinline__ void cp_async_group_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/// Blocks until all previous cp.async.commit_group operations have committed.
// cp.async.wait_all is equivalent to :
// cp.async.commit_group;
// cp.async.wait_group 0;
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

#endif
