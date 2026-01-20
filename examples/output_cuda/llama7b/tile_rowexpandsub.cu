// PTO Program: tile_rowexpandsub
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_rowexpandsub
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 32,896 bytes (32.1 KB)
//   Total capacity (w/ reuse): 32,896 bytes (32.1 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   result               32x128     f32     16384   [  2,   3]           -
//   row_vals             32x1       f32       128   [  1,   2]           -
//   x                    32x128     f32     16384   [  0,   2]           -
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

__device__ float x[32][128];
__device__ float row_vals[32][1];
__device__ float result[32][128];

__global__ void tile_rowexpandsub_kernel(float* input_x, float* input_row, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 32 && _col < 128) {
        x[_row][_col] = input_x[_row * 128 + _col];
    }

    // FUSED (1 ops): row_vals=TLOAD(...)
    if (_row < 32 && _col < 1) {
        row_vals[_row][_col] = input_row[_row * 1 + _col];
    }

    // FUSED (2 ops): result=TROWEXPANDSUB(...); output=TSTORE(...)
    if (_row < 32 && _col < 128) {
        result[_row][_col] = x[_row][_col] - row_vals[_row][0];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void tile_rowexpandsub(float* input_x, float* input_row, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    tile_rowexpandsub_kernel<<<grid, block>>>(input_x, input_row, output);
    cudaDeviceSynchronize();
}