// PTO Program: rope_tile
// Function Type: InCore (tile-level computation)
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void rope_tile(float* input, float* cos_cache, float* sin_cache, float* output) {
    float x[8][8];
    float cos_pos[8][8];
    float sin_pos[8][8];
    float x_cos[8][8];
    float x_sin[8][8];
    float result[8][8];

    // Loop fusion: 6 loop overheads saved

    // FUSED LOOP (7 ops): x=TLOAD(input,0,0); cos_pos=TLOAD(cos_cache,0,0); sin_pos=TLOAD(sin_cache,0,0); x_cos=TMUL(x,cos_pos); x_sin=TMUL(x,sin_pos); result=TADD(x_cos,x_sin); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&cos_cache[_row * 8 + _col]);
            vst1q_f32(&cos_pos[_row][_col], _vl1);
            float32x4_t _vl2 = vld1q_f32(&sin_cache[_row * 8 + _col]);
            vst1q_f32(&sin_pos[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v4 = vld1q_f32(&cos_pos[_row][_col]);
            float32x4_t _vr5 = vmulq_f32(_v3, _v4);
            vst1q_f32(&x_cos[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v7 = vld1q_f32(&sin_pos[_row][_col]);
            float32x4_t _vr8 = vmulq_f32(_v6, _v7);
            vst1q_f32(&x_sin[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&x_cos[_row][_col]);
            float32x4_t _v10 = vld1q_f32(&x_sin[_row][_col]);
            float32x4_t _vr11 = vaddq_f32(_v9, _v10);
            vst1q_f32(&result[_row][_col], _vr11);
            float32x4_t _vs12 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs12);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
            cos_pos[_row][_col] = cos_cache[_row * 8 + _col];
            sin_pos[_row][_col] = sin_cache[_row * 8 + _col];
            x_cos[_row][_col] = x[_row][_col] * cos_pos[_row][_col];
            x_sin[_row][_col] = x[_row][_col] * sin_pos[_row][_col];
            result[_row][_col] = x_cos[_row][_col] + x_sin[_row][_col];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}