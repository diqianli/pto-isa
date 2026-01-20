// PTO Program: direct_fused_softmax
// Function Type: InCore (tile-level computation)
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

__device__ float x[8][8];
__device__ float row_max[8][1];
__device__ float x_shifted[8][8];
__device__ float exp_x[8][8];
__device__ float row_sum[8][1];
__device__ float result[8][8];

__global__ void direct_fused_softmax_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 2 loop overheads saved

    // FUSED (1 ops): x=TLOAD(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
    }

    // TROWMAX: row_max = rowmax(x)
    if (_col == 0 && _row < 8) {
        float _max = x[_row][0];
        for (int _c = 1; _c < 8; _c++) if (x[_row][_c] > _max) _max = x[_row][_c];
        row_max[_row][0] = _max;}

    // FUSED (2 ops): x_shifted=TROWEXPANDSUB(...); exp_x=TEXP(...)
    if (_row < 8 && _col < 8) {
        x_shifted[_row][_col] = x[_row][_col] - row_max[_row][0];
        exp_x[_row][_col] = __expf(x_shifted[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(exp_x)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += exp_x[_row][_c];
        row_sum[_row][0] = _sum;}

    // FUSED (2 ops): result=TROWEXPANDDIV(...); output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        result[_row][_col] = exp_x[_row][_col] / row_sum[_row][0];
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void direct_fused_softmax(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    direct_fused_softmax_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}