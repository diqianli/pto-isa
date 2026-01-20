// PTO Program: prims_log
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void prims_log(float* input, float* output) {
    float x[1][4096];
    float result[1][4096];

    // Loop fusion: 5 loop overheads saved

    // FUSED LOOP (6 ops): x=TLOAD(input,0,0); result=TLOG(x); output=TSTORE(result,0,0); x=TLOAD(input,0,0); result=TLOG(x); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&result[_row][_col], _vr2);
            float32x4_t _vs3 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs3);
            float32x4_t _vl4 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl4);
            float32x4_t _v5 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr6 = _v5;
            vst1q_f32(&result[_row][_col], _vr6);
            float32x4_t _vs7 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs7);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input[_row * 4096 + _col];
            result[_row][_col] = logf(x[_row][_col]);
            output[_row * 4096 + _col] = result[_row][_col];
            x[_row][_col] = input[_row * 4096 + _col];
            result[_row][_col] = logf(x[_row][_col]);
            output[_row * 4096 + _col] = result[_row][_col];
        }
    }

}