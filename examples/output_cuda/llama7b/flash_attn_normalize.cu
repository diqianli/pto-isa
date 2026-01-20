// PTO Program: flash_attn_normalize
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_normalize
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 65,792 bytes (64.2 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_vec                64x1       f32       256   [  1,   2]           -
//   o_block              64x128     f32     32768   [  0,   2]           -
//   o_final              64x128     f32     32768   [  2,   3]           -
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

__device__ float o_block[64][128];
__device__ float l_vec[64][1];
__device__ float o_final[64][128];

__global__ void flash_attn_normalize_kernel(float* input_o, float* input_l, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): o_block=TLOAD(...)
    if (_row < 64 && _col < 128) {
        o_block[_row][_col] = input_o[_row * 128 + _col];
    }

    // FUSED (1 ops): l_vec=TLOAD(...)
    if (_row < 64 && _col < 1) {
        l_vec[_row][_col] = input_l[_row * 1 + _col];
    }

    // FUSED (2 ops): o_final=TROWEXPANDDIV(...); output=TSTORE(...)
    if (_row < 64 && _col < 128) {
        o_final[_row][_col] = o_block[_row][_col] / l_vec[_row][0];
        output[_row * 128 + _col] = o_final[_row][_col];
    }

}

void flash_attn_normalize(float* input_o, float* input_l, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    flash_attn_normalize_kernel<<<grid, block>>>(input_o, input_l, output);
    cudaDeviceSynchronize();
}