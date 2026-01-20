// PTO Program: aten_mul_scalar
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_mul_scalar(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    float x[1][4096];
    float result[1][4096];

    // Loop fusion: 4 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (3 ops): x=TLOAD(input,tile_idx,0); result=TMULS(x,0.5f); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs0 = vdupq_n_f32(0.5f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl1 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl1);
                float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr3 = vmulq_f32(_v2, _vs0);
                vst1q_f32(&result[_row][_col], _vr3);
                float32x4_t _vs4 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs4);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
                result[_row][_col] = x[_row][_col] * 0.5f;
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (3 ops): x=TLOAD(input,num_full_tiles,0); result=TMULS(x,0.5f); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs5 = vdupq_n_f32(0.5f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl6 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl6);
                float32x4_t _v7 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr8 = vmulq_f32(_v7, _vs5);
                vst1q_f32(&result[_row][_col], _vr8);
                float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs9);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
                result[_row][_col] = x[_row][_col] * 0.5f;
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}