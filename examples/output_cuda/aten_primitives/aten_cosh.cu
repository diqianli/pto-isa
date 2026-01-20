// PTO Program: aten_cosh
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
__device__ float neg_x[1][4096];
__device__ float exp_x[1][4096];
__device__ float exp_neg_x[1][4096];
__device__ float sum[1][4096];
__device__ float result[1][4096];

__global__ void aten_cosh_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 13 loop overheads saved

    // FUSED (14 ops): x=TLOAD(...); exp_x=TEXP(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); sum=TADD(...); result=TDIVS(...); output=TSTORE(...); x=TLOAD(...); exp_x=TEXP(...); neg_x=TNEG(...); exp_neg_x=TEXP(...); sum=TADD(...); result=TDIVS(...); output=TSTORE(...)
    if (_row < 1 && _col < 4096) {
        x[_row][_col] = input[_row * 4096 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        neg_x[_row][_col] = -x[_row][_col];
        exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
        sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
        result[_row][_col] = sum[_row][_col] / 2.0f;
        output[_row * 4096 + _col] = result[_row][_col];
        x[_row][_col] = input[_row * 4096 + _col];
        exp_x[_row][_col] = __expf(x[_row][_col]);
        neg_x[_row][_col] = -x[_row][_col];
        exp_neg_x[_row][_col] = __expf(neg_x[_row][_col]);
        sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
        result[_row][_col] = sum[_row][_col] / 2.0f;
        output[_row * 4096 + _col] = result[_row][_col];
    }

}

void aten_cosh(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_cosh_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}