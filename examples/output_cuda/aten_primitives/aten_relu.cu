// PTO Program: aten_relu
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
__device__ float result[1][4096];

__global__ void aten_relu_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 5 loop overheads saved

    // FUSED (6 ops): x=TLOAD(...); result=TRELU(...); output=TSTORE(...); x=TLOAD(...); result=TRELU(...); output=TSTORE(...)
    if (_row < 1 && _col < 4096) {
        x[_row][_col] = input[_row * 4096 + _col];
        result[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        output[_row * 4096 + _col] = result[_row][_col];
        x[_row][_col] = input[_row * 4096 + _col];
        result[_row][_col] = fmaxf(x[_row][_col], 0.0f);
        output[_row * 4096 + _col] = result[_row][_col];
    }

}

void aten_relu(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_relu_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}