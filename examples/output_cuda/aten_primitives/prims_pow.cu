// PTO Program: prims_pow
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

__device__ float base[1][4096];
__device__ float exp_tile[1][4096];
__device__ float log_base[1][4096];
__device__ float product[1][4096];
__device__ float result[1][4096];

__global__ void prims_pow_kernel(float* input_base, float* input_exp, float* output) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 11 loop overheads saved

    // FUSED (12 ops): base=TLOAD(...); exp_tile=TLOAD(...); log_base=TLOG(...); product=TMUL(...); result=TEXP(...); output=TSTORE(...); base=TLOAD(...); exp_tile=TLOAD(...); log_base=TLOG(...); product=TMUL(...); result=TEXP(...); output=TSTORE(...)
    if (_row < 1 && _col < 4096) {
        base[_row][_col] = input_base[_row * 4096 + _col];
        exp_tile[_row][_col] = input_exp[_row * 4096 + _col];
        log_base[_row][_col] = __logf(base[_row][_col]);
        product[_row][_col] = exp_tile[_row][_col] * log_base[_row][_col];
        result[_row][_col] = __expf(product[_row][_col]);
        output[_row * 4096 + _col] = result[_row][_col];
        base[_row][_col] = input_base[_row * 4096 + _col];
        exp_tile[_row][_col] = input_exp[_row * 4096 + _col];
        log_base[_row][_col] = __logf(base[_row][_col]);
        product[_row][_col] = exp_tile[_row][_col] * log_base[_row][_col];
        result[_row][_col] = __expf(product[_row][_col]);
        output[_row * 4096 + _col] = result[_row][_col];
    }

}

void prims_pow(float* input_base, float* input_exp, float* output) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_pow_kernel<<<grid, block>>>(input_base, input_exp, output);
    cudaDeviceSynchronize();
}