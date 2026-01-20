// PTO Program: flash_attn_score_block
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_score_block
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 98,304 bytes (96.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   k_block              64x128     f32     32768   [  1,  -1]           -
//   q_block              64x128     f32     32768   [  0,  -1]           -
//   s_block              64x64      f32     16384   [  2,   4]           -
//   s_scaled             64x64      f32     16384   [  4,   5]           -
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

__device__ float q_block[64][128];
__device__ float k_block[64][128];
__device__ float s_block[64][64];
__device__ float s_scaled[64][64];

__global__ void flash_attn_score_block_kernel(float* input_q, float* input_k, float* output_s) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (2 ops): q_block=TLOAD(...); k_block=TLOAD(...)
    if (_row < 64 && _col < 128) {
        q_block[_row][_col] = input_q[_row * 128 + _col];
        k_block[_row][_col] = input_k[_row * 128 + _col];
    }

    // TMATMUL: s_block = q_block @ k_block
    if (_row < 64 && _col < 64) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 128; _k++) _sum += q_block[_row][_k] * k_block[_k][_col];
        s_block[_row][_col] = _sum;}

    int scale = 0.08838834764831843;

    // FUSED (2 ops): s_scaled=TMULS(...); output_s=TSTORE(...)
    if (_row < 64 && _col < 64) {
        s_scaled[_row][_col] = s_block[_row][_col] * scalef;
        output_s[_row * 64 + _col] = s_scaled[_row][_col];
    }

}

void flash_attn_score_block(float* input_q, float* input_k, float* output_s) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    flash_attn_score_block_kernel<<<grid, block>>>(input_q, input_k, output_s, scale);
    cudaDeviceSynchronize();
}