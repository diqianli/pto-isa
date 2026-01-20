// PTO Program: F_l1_loss
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

__device__ float pred[8][8];
__device__ float target[8][8];
__device__ float diff[8][8];
__device__ float abs_diff[8][8];
__device__ float row_sum[8][1];
__device__ float result[1][1];

__global__ void F_l1_loss_kernel(float* input, float* target_mem, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 4 loop overheads saved

    // FUSED (4 ops): pred=TLOAD(...); target=TLOAD(...); diff=TSUB(...); abs_diff=TABS(...)
    if (_row < 8 && _col < 8) {
        pred[_row][_col] = input[_row * 8 + _col];
        target[_row][_col] = target_mem[_row * 8 + _col];
        diff[_row][_col] = pred[_row][_col] - target[_row][_col];
        abs_diff[_row][_col] = fabsf(diff[_row][_col]);
    }

    // TROWSUM: row_sum = rowsum(abs_diff)
    if (_col == 0 && _row < 8) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 8; _c++) _sum += abs_diff[_row][_c];
        row_sum[_row][0] = _sum;}

    // TCOLSUM: Not implemented

    // FUSED (2 ops): result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 1) {
        result[_row][_col] = result[_row][_col] / 64.0f;
        output[_row * 1 + _col] = result[_row][_col];
    }

}

void F_l1_loss(float* input, float* target_mem, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    F_l1_loss_kernel<<<grid, block>>>(input, target_mem, output);
    cudaDeviceSynchronize();
}