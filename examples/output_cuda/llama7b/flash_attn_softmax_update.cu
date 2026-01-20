// PTO Program: flash_attn_softmax_update
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_softmax_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 51,456 bytes (50.2 KB)
//   Total capacity (w/ reuse): 34,048 bytes (33.2 KB)
//   Reuse savings:            17,408 bytes (33.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_new                64x1       f32       256   [ 11,  13]           -
//   l_prev               64x1       f32       256   [  2,   9]           -
//   l_scaled             64x1       f32       256   [  9,  11]           <- m_diff
//   m_cur                64x1       f32       256   [  3,   4]           -
//   m_diff               64x1       f32       256   [  7,   8]           <- m_cur
//   m_new                64x1       f32       256   [  4,  12]           -
//   m_prev               64x1       f32       256   [  1,   7]           -
//   p_block              64x64      f32     16384   [  6,  14]           <- s_block
//   p_rowsum             64x1       f32       256   [ 10,  11]           <- l_prev
//   s_block              64x64      f32     16384   [  0,   5]           -
//   s_shifted            64x64      f32     16384   [  5,   6]           -
//   scale_old            64x1       f32       256   [  8,  15]           <- m_prev
//
// BUFFER REUSE MAP:
//   p_block reuses buffer of s_block
//   scale_old reuses buffer of m_prev
//   m_diff reuses buffer of m_cur
//   l_scaled reuses buffer of m_diff
//   p_rowsum reuses buffer of l_prev
//
// ======================================================================

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

__device__ float s_block[64][64];
__device__ float m_prev[64][1];
__device__ float l_prev[64][1];
__device__ float m_new[64][1];
__device__ float m_cur[64][1];
__device__ float l_new[64][1];
__device__ float p_block[64][64];
__device__ float s_shifted[64][64];
__device__ float scale_old[64][1];
__device__ float m_diff[64][1];
__device__ float l_scaled[64][1];
__device__ float p_rowsum[64][1];

__global__ void flash_attn_softmax_update_kernel(float* input_s, float* input_m_prev, float* input_l_prev, float* output_m_new, float* output_l_new, float* output_p, float* output_scale_old) {
    int _row = threadIdx.y + blockIdx.y * blockDim.y;
    int _col = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop fusion: 6 loop overheads saved

    // FUSED (1 ops): s_block=TLOAD(...)
    if (_row < 64 && _col < 64) {
        s_block[_row][_col] = input_s[_row * 64 + _col];
    }

    // FUSED (2 ops): m_prev=TLOAD(...); l_prev=TLOAD(...)
    if (_row < 64 && _col < 1) {
        m_prev[_row][_col] = input_m_prev[_row * 1 + _col];
        l_prev[_row][_col] = input_l_prev[_row * 1 + _col];
    }

    // TROWMAX: m_cur = rowmax(s_block)
    if (_col == 0 && _row < 64) {
        float _max = s_block[_row][0];
        for (int _c = 1; _c < 64; _c++) if (s_block[_row][_c] > _max) _max = s_block[_row][_c];
        m_cur[_row][0] = _max;}

    // FUSED (1 ops): m_new=TMAX(...)
    if (_row < 64 && _col < 1) {
        m_new[_row][_col] = fmaxf(m_prev[_row][_col], m_cur[_row][_col]);
    }

    // FUSED (2 ops): s_shifted=TROWEXPANDSUB(...); p_block=TEXP(...)
    if (_row < 64 && _col < 64) {
        s_shifted[_row][_col] = s_block[_row][_col] - m_new[_row][0];
        p_block[_row][_col] = __expf(s_shifted[_row][_col]);
    }

    // FUSED (3 ops): m_diff=TSUB(...); scale_old=TEXP(...); l_scaled=TMUL(...)
    if (_row < 64 && _col < 1) {
        m_diff[_row][_col] = m_prev[_row][_col] - m_new[_row][_col];
        scale_old[_row][_col] = __expf(m_diff[_row][_col]);
        l_scaled[_row][_col] = scale_old[_row][_col] * l_prev[_row][_col];
    }

    // TROWSUM: p_rowsum = rowsum(p_block)
    if (_col == 0 && _row < 64) {
        float _sum = 0.0f;
        for (int _c = 0; _c < 64; _c++) _sum += p_block[_row][_c];
        p_rowsum[_row][0] = _sum;}

    // FUSED (3 ops): l_new=TADD(...); output_m_new=TSTORE(...); output_l_new=TSTORE(...)
    if (_row < 64 && _col < 1) {
        l_new[_row][_col] = l_scaled[_row][_col] + p_rowsum[_row][_col];
        output_m_new[_row * 1 + _col] = m_new[_row][_col];
        output_l_new[_row * 1 + _col] = l_new[_row][_col];
    }

    // FUSED (1 ops): output_p=TSTORE(...)
    if (_row < 64 && _col < 64) {
        output_p[_row * 64 + _col] = p_block[_row][_col];
    }

    // FUSED (1 ops): output_scale_old=TSTORE(...)
    if (_row < 64 && _col < 1) {
        output_scale_old[_row * 1 + _col] = scale_old[_row][_col];
    }

}

void flash_attn_softmax_update(float* input_s, float* input_m_prev, float* input_l_prev, float* output_m_new, float* output_l_new, float* output_p, float* output_scale_old) {
    dim3 block(8, 8);
    dim3 grid(1, 1);
    flash_attn_softmax_update_kernel<<<grid, block>>>(input_s, input_m_prev, input_l_prev, output_m_new, output_l_new, output_p, output_scale_old);
    cudaDeviceSynchronize();
}