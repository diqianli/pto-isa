// PTO Program: tile_muls
// Function Type: InCore (tile-level computation)
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tile_muls(float* input, float* output, float scale) {
    float a[8][8];
    float result[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (3 ops): a=TLOAD(input,0,0); result=TMULS(a,scalef); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(scalef);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&a[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&a[_row][_col]);
            float32x4_t _vr3 = vmulq_f32(_v2, _vs0);
            vst1q_f32(&result[_row][_col], _vr3);
            float32x4_t _vs4 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs4);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            a[_row][_col] = input[_row * 8 + _col];
            result[_row][_col] = a[_row][_col] * scalef;
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}