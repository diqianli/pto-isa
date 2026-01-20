// PTO Program: sinh_taylor
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
__device__ float x_squared[32][128];
__device__ float term[32][128];
__device__ float result[32][128];

__global__ void sinh_taylor_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 45 loop overheads saved

    // FUSED (46 ops): x=TLOAD(...); result=TMULS(...); x_squared=TMUL(...); term=TMULS(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); output=TSTORE(...); x=TLOAD(...); result=TMULS(...); x_squared=TMUL(...); term=TMULS(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); term=TMUL(...); term=TDIVS(...); result=TADD(...); output=TSTORE(...)
    if (_row < 32 && _col < 128) {
        x[_row][_col] = input[_row * 128 + _col];
        result[_row][_col] = x[_row][_col] * 1.0f;
        x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
        term[_row][_col] = x[_row][_col] * 1.0f;
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 6.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 20.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 42.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 72.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 110.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 156.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
        x[_row][_col] = input[_row * 128 + _col];
        result[_row][_col] = x[_row][_col] * 1.0f;
        x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
        term[_row][_col] = x[_row][_col] * 1.0f;
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 6.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 20.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 42.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 72.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 110.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 156.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        output[_row * 128 + _col] = result[_row][_col];
    }

}

void sinh_taylor(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    sinh_taylor_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}