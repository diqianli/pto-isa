// PTO Program: prims_maximum
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

__device__ float x[1][4096];
__device__ float y[1][4096];
__device__ float result[1][4096];

__global__ void prims_maximum_kernel(float* input_x, float* input_y, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 7 loop overheads saved

    // FUSED (8 ops): x=TLOAD(...); y=TLOAD(...); result=TMAX(...); output=TSTORE(...); x=TLOAD(...); y=TLOAD(...); result=TMAX(...); output=TSTORE(...)
    if (_row < 1 && _col < 4096) {
        x[_row][_col] = input_x[_row * 4096 + _col];
        y[_row][_col] = input_y[_row * 4096 + _col];
        result[_row][_col] = fmaxf(x[_row][_col], y[_row][_col]);
        output[_row * 4096 + _col] = result[_row][_col];
        x[_row][_col] = input_x[_row * 4096 + _col];
        y[_row][_col] = input_y[_row * 4096 + _col];
        result[_row][_col] = fmaxf(x[_row][_col], y[_row][_col]);
        output[_row * 4096 + _col] = result[_row][_col];
    }

}

void prims_maximum(float* input_x, float* input_y, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_maximum_kernel<<<grid, block>>>(input_x, input_y, output);
    cudaDeviceSynchronize();
}