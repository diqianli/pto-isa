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

__global__ void prims_pow_kernel(float* input_base, float* input_exp, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 10 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (6 ops): base=TLOAD(...); exp_tile=TLOAD(...); log_base=TLOG(...); product=TMUL(...); result=TEXP(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            base[_row][_col] = input_base[(tile_idx) * 4096 + _row * 4096 + _col];
            exp_tile[_row][_col] = input_exp[(tile_idx) * 4096 + _row * 4096 + _col];
            log_base[_row][_col] = __logf(base[_row][_col]);
            product[_row][_col] = exp_tile[_row][_col] * log_base[_row][_col];
            result[_row][_col] = __expf(product[_row][_col]);
            output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (6 ops): base=TLOAD(...); exp_tile=TLOAD(...); log_base=TLOG(...); product=TMUL(...); result=TEXP(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            base[_row][_col] = input_base[(num_full_tiles) * 4096 + _row * 4096 + _col];
            exp_tile[_row][_col] = input_exp[(num_full_tiles) * 4096 + _row * 4096 + _col];
            log_base[_row][_col] = __logf(base[_row][_col]);
            product[_row][_col] = exp_tile[_row][_col] * log_base[_row][_col];
            result[_row][_col] = __expf(product[_row][_col]);
            output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

}

void prims_pow(float* input_base, float* input_exp, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_pow_kernel<<<grid, block>>>(input_base, input_exp, output, num_full_tiles, tail_elements, zero, tile_size);
    cudaDeviceSynchronize();
}