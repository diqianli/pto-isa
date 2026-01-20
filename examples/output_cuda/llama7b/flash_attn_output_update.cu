// PTO Program: flash_attn_output_update
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_output_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 180,480 bytes (176.2 KB)
//   Total capacity (w/ reuse): 147,712 bytes (144.2 KB)
//   Reuse savings:            32,768 bytes (18.2%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   o_new                64x128     f32     32768   [  6,   7]           -
//   o_prev               64x128     f32     32768   [  0,   4]           -
//   o_scaled             64x128     f32     32768   [  4,   6]           -
//   p_block              64x64      f32     16384   [  1,  -1]           -
//   pv                   64x128     f32     32768   [  5,   6]           <- o_prev
//   scale_old            64x1       f32       256   [  3,   4]           -
//   v_block              64x128     f32     32768   [  2,  -1]           -
//
// BUFFER REUSE MAP:
//   pv reuses buffer of o_prev
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

__device__ float o_prev[64][128];
__device__ float p_block[64][64];
__device__ float v_block[64][128];
__device__ float scale_old[64][1];
__device__ float o_scaled[64][128];
__device__ float pv[64][128];
__device__ float o_new[64][128];

__global__ void flash_attn_output_update_kernel(float* input_o_prev, float* input_p, float* input_v, float* input_scale, float* output_o) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): o_prev=TLOAD(...)
    if (_row < 64 && _col < 128) {
        o_prev[_row][_col] = input_o_prev[_row * 128 + _col];
    }

    // FUSED (1 ops): p_block=TLOAD(...)
    if (_row < 64 && _col < 64) {
        p_block[_row][_col] = input_p[_row * 64 + _col];
    }

    // FUSED (1 ops): v_block=TLOAD(...)
    if (_row < 64 && _col < 128) {
        v_block[_row][_col] = input_v[_row * 128 + _col];
    }

    // FUSED (1 ops): scale_old=TLOAD(...)
    if (_row < 64 && _col < 1) {
        scale_old[_row][_col] = input_scale[_row * 1 + _col];
    }

    // FUSED (1 ops): o_scaled=TROWEXPANDMUL(...)
    if (_row < 64 && _col < 128) {
        o_scaled[_row][_col] = o_prev[_row][_col] * scale_old[_row][0];
    }

    // TMATMUL: pv = p_block @ v_block
    if (_row < 64 && _col < 128) {
        float _sum = 0.0f;
        for (int _k = 0; _k < 64; _k++) _sum += p_block[_row][_k] * v_block[_k][_col];
        pv[_row][_col] = _sum;}

    // FUSED (2 ops): o_new=TADD(...); output_o=TSTORE(...)
    if (_row < 64 && _col < 128) {
        o_new[_row][_col] = o_scaled[_row][_col] + pv[_row][_col];
        output_o[_row * 128 + _col] = o_new[_row][_col];
    }

}

void flash_attn_output_update(float* input_o_prev, float* input_p, float* input_v, float* input_scale, float* output_o) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    flash_attn_output_update_kernel<<<grid, block>>>(input_o_prev, input_p, input_v, input_scale, output_o);
    cudaDeviceSynchronize();
}