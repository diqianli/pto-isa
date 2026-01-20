// PTO Program: aten_mean_row
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_mean_row(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    float x[1][4096];
    float sum_result[1][1];
    float result[1][1];

    // Loop fusion: 2 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (1 ops): x=TLOAD(input,tile_idx,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl0 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl0);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
            }
        }

        // TROWSUM: sum_result = rowsum(x)
        for (int _row = 0; _row < 1; _row++) {
            float _sum = 0.0f;
            for (int _col = 0; _col < 4096; _col++) {
                _sum += x[_row][_col];
            }
            sum_result[_row][0] = _sum;}

        // FUSED LOOP (2 ops): result=TDIVS(sum_result,4096.0f); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs1 = vdupq_n_f32(4096.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 1; _col += 4) {
                float32x4_t _v2 = vld1q_f32(&sum_result[_row][_col]);
                float32x4_t _vr3 = vdivq_f32(_v2, _vs1);
                vst1q_f32(&result[_row][_col], _vr3);
                float32x4_t _vs4 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 1 + _row * 1 + _col], _vs4);
            }
            // Scalar cleanup
            for (; _col < 1; _col++) {
                result[_row][_col] = sum_result[_row][_col] / 4096.0f;
                output[(tile_idx) * 1 + _row * 1 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (1 ops): x=TLOAD(input,num_full_tiles,0)
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl5 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl5);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
            }
        }

        // TROWSUM: sum_result = rowsum(x)
        for (int _row = 0; _row < 1; _row++) {
            float _sum = 0.0f;
            for (int _col = 0; _col < 4096; _col++) {
                _sum += x[_row][_col];
            }
            sum_result[_row][0] = _sum;}

        // FUSED LOOP (2 ops): result=TDIVS(sum_result,4096.0f); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs6 = vdupq_n_f32(4096.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 1; _col += 4) {
                float32x4_t _v7 = vld1q_f32(&sum_result[_row][_col]);
                float32x4_t _vr8 = vdivq_f32(_v7, _vs6);
                vst1q_f32(&result[_row][_col], _vr8);
                float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 1 + _row * 1 + _col], _vs9);
            }
            // Scalar cleanup
            for (; _col < 1; _col++) {
                result[_row][_col] = sum_result[_row][_col] / 4096.0f;
                output[(num_full_tiles) * 1 + _row * 1 + _col] = result[_row][_col];
            }
        }

    }

}