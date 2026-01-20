// PTO Program: nn_LogSoftmax
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
__device__ float exp_x[8][8];
__device__ float sum_exp[8][1];
__device__ float log_sum[8][1];
__device__ float result[8][8];

__global__ void nn_LogSoftmax_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 1 loop overheads saved

    // FUSED (2 ops): x=TLOAD(...); exp_x=TEXP(...)
    if (_row < 8 && _col < 8) {
        x[_row][_col] = input[_row * 8 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
    }

    // TROWSUM: sum_exp = rowsum(exp_x)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += exp_x[_row][_c];
        sum_exp[_row][0] = _sum;}

    // FUSED (1 ops): log_sum=TLOG(...)
    if (_row < 8 && _col < 1) {
        log_sum[_row][_col] = __logf(sum_exp[_row][_col]);
    }

    // TROWEXPANDSUB: Not implemented

    // FUSED (1 ops): output=TSTORE(...)
    if (_row < 8 && _col < 8) {
        output[_row * 8 + _col] = result[_row][_col];
    }

}

void nn_LogSoftmax(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    nn_LogSoftmax_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}