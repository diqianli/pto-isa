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

__global__ void prims_maximum_kernel(float* input_x, float* input_y, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED (4 ops): x=TLOAD(...); y=TLOAD(...); result=TMAX(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input_x[(tile_idx) * 4096 + _row * 4096 + _col];
            y[_row][_col] = input_y[(tile_idx) * 4096 + _row * 4096 + _col];
            result[_row][_col] = fmaxf(x[_row][_col], y[_row][_col]);
            output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED (4 ops): x=TLOAD(...); y=TLOAD(...); result=TMAX(...); output=TSTORE(...)
        if (_row < 1 && _col < 4096) {
            x[_row][_col] = input_x[(num_full_tiles) * 4096 + _row * 4096 + _col];
            y[_row][_col] = input_y[(num_full_tiles) * 4096 + _row * 4096 + _col];
            result[_row][_col] = fmaxf(x[_row][_col], y[_row][_col]);
            output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
        }

    }

}

void prims_maximum(float* input_x, float* input_y, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    prims_maximum_kernel<<<grid, block>>>(input_x, input_y, output, num_full_tiles, tail_elements, zero, tile_size);
    cudaDeviceSynchronize();
}