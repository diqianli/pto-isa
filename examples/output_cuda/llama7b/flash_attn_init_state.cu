// PTO Program: flash_attn_init_state
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_init_state
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 33,280 bytes (32.5 KB)
//   Total capacity (w/ reuse): 33,280 bytes (32.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_init               64x1       f32       256   [  1,   4]           -
//   m_init               64x1       f32       256   [  2,   5]           -
//   o_init               64x128     f32     32768   [  0,   3]           -
//
// ======================================================================

// Auto-generated CUDA code from PTO ISA Compiler
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__device__ float o_init[64][128];
__device__ float l_init[64][1];
__device__ float m_init[64][1];

__global__ void flash_attn_init_state_kernel(float* input_zeros_large, float* input_zeros_small, float* input_neg_inf, float* output_o, float* output_l, float* output_m) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (1 ops): o_init=TLOAD(...)
    if (_row < 64 && _col < 128) {
        o_init[_row][_col] = input_zeros_large[_row * 128 + _col];
    }

    // FUSED (2 ops): l_init=TLOAD(...); m_init=TLOAD(...)
    if (_row < 64 && _col < 1) {
        l_init[_row][_col] = input_zeros_small[_row * 1 + _col];
        m_init[_row][_col] = input_neg_inf[_row * 1 + _col];
    }

    // FUSED (1 ops): output_o=TSTORE(...)
    if (_row < 64 && _col < 128) {
        output_o[_row * 128 + _col] = o_init[_row][_col];
    }

    // FUSED (2 ops): output_l=TSTORE(...); output_m=TSTORE(...)
    if (_row < 64 && _col < 1) {
        output_l[_row * 1 + _col] = l_init[_row][_col];
        output_m[_row * 1 + _col] = m_init[_row][_col];
    }

}

void flash_attn_init_state(float* input_zeros_large, float* input_zeros_small, float* input_neg_inf, float* output_o, float* output_l, float* output_m) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    flash_attn_init_state_kernel<<<grid, block>>>(input_zeros_large, input_zeros_small, input_neg_inf, output_o, output_l, output_m);
    cudaDeviceSynchronize();
}