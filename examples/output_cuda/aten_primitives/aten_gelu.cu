// PTO Program: aten_gelu
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
__device__ float scaled_x[1][4096];
__device__ float neg_scaled[1][4096];
__device__ float exp_neg[1][4096];
__device__ float one_plus[1][4096];
__device__ float sigmoid_out[1][4096];
__device__ float result[1][4096];

__global__ void aten_gelu_kernel(float* input, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 15 loop overheads saved

    // FUSED (16 ops): x=TLOAD(...); scaled_x=TMULS(...); neg_scaled=TNEG(...); exp_neg=TEXP(...); one_plus=TADDS(...); sigmoid_out=TRECIP(...); result=TMUL(...); output=TSTORE(...); x=TLOAD(...); scaled_x=TMULS(...); neg_scaled=TNEG(...); exp_neg=TEXP(...); one_plus=TADDS(...); sigmoid_out=TRECIP(...); result=TMUL(...); output=TSTORE(...)
    if (_row < 1 && _col < 4096) {
        x[_row][_col] = input[_row * 4096 + _col];
        scaled_x[_row][_col] = x[_row][_col] * 1.702f;
        neg_scaled[_row][_col] = -scaled_x[_row][_col];
        exp_neg[_row][_col] = __expf(neg_scaled[_row][_col]);
        one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
        sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
        result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
        output[_row * 4096 + _col] = result[_row][_col];
        x[_row][_col] = input[_row * 4096 + _col];
        scaled_x[_row][_col] = x[_row][_col] * 1.702f;
        neg_scaled[_row][_col] = -scaled_x[_row][_col];
        exp_neg[_row][_col] = __expf(neg_scaled[_row][_col]);
        one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
        sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
        result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
        output[_row * 4096 + _col] = result[_row][_col];
    }

}

void aten_gelu(float* input, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    aten_gelu_kernel<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}