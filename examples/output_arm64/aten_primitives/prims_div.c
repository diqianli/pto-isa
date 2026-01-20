// PTO Program: prims_div
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void prims_div(float* input_x, float* input_y, float* output) {
    float x[1][4096];
    float y[1][4096];
    float result[1][4096];

    // Loop fusion: 7 loop overheads saved

    // FUSED LOOP (8 ops): x=TLOAD(input_x,0,0); y=TLOAD(input_y,0,0); result=TDIV(x,y); output=TSTORE(result,0,0); x=TLOAD(input_x,0,0); y=TLOAD(input_y,0,0); result=TDIV(x,y); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_x[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_y[_row * 4096 + _col]);
            vst1q_f32(&y[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v3 = vld1q_f32(&y[_row][_col]);
            float32x4_t _vr4 = vdivq_f32(_v2, _v3);
            vst1q_f32(&result[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs5);
            float32x4_t _vl6 = vld1q_f32(&input_x[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl6);
            float32x4_t _vl7 = vld1q_f32(&input_y[_row * 4096 + _col]);
            vst1q_f32(&y[_row][_col], _vl7);
            float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&y[_row][_col]);
            float32x4_t _vr10 = vdivq_f32(_v8, _v9);
            vst1q_f32(&result[_row][_col], _vr10);
            float32x4_t _vs11 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs11);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input_x[_row * 4096 + _col];
            y[_row][_col] = input_y[_row * 4096 + _col];
            result[_row][_col] = x[_row][_col] / y[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
            x[_row][_col] = input_x[_row * 4096 + _col];
            y[_row][_col] = input_y[_row * 4096 + _col];
            result[_row][_col] = x[_row][_col] / y[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
        }
    }

}