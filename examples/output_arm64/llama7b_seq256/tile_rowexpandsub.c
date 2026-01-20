// PTO Program: tile_rowexpandsub
// Function Type: InCore (tile-level computation)
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void tile_rowexpandsub(float* input_x, float* input_row, float* output) {
    float x[8][8];
    float row_vals[8][1];
    float result[8][8];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input_x,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_x[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input_x[_row * 8 + _col];
        }
    }

    // FUSED LOOP (1 ops): row_vals=TLOAD(input_row,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input_row[_row * 1 + _col]);
            vst1q_f32(&row_vals[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            row_vals[_row][_col] = input_row[_row * 1 + _col];
        }
    }

    // FUSED LOOP (2 ops): result=TROWEXPANDSUB(x,row_vals); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v02 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb4 = vdupq_n_f32(row_vals[_row][0]);
            float32x4_t _vr3 = vsubq_f32(_v02, _vb4);
            vst1q_f32(&result[_row][_col], _vr3);
            float32x4_t _vs5 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] - row_vals[_row][0];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}