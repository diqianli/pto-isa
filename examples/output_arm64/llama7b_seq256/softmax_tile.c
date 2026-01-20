// PTO Program: softmax_tile
// Function Type: InCore (tile-level computation)
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void softmax_tile(float* input, float* output) {
    float x[8][8];
    float row_max[8][1];
    float x_shifted[8][8];
    float exp_x[8][8];
    float row_sum[8][1];
    float result[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 8 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x[_row][_col] = input[_row * 8 + _col];
        }
    }

    // TROWMAX: row_max = rowmax(x)
    for (int _row = 0; _row < 8; _row++) {
        float _max = x[_row][0];
        for (int _col = 1; _col < 8; _col++) {
            if (x[_row][_col] > _max) _max = x[_row][_col];
        }
        row_max[_row][0] = _max;}

    // FUSED LOOP (2 ops): x_shifted=TROWEXPANDSUB(x,row_max); exp_x=TEXP(x_shifted)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v01 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vb3 = vdupq_n_f32(row_max[_row][0]);
            float32x4_t _vr2 = vsubq_f32(_v01, _vb3);
            vst1q_f32(&x_shifted[_row][_col], _vr2);
            float32x4_t _v4 = vld1q_f32(&x_shifted[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&exp_x[_row][_col], _vr5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            x_shifted[_row][_col] = x[_row][_col] - row_max[_row][0];
            exp_x[_row][_col] = expf(x_shifted[_row][_col]);
        }
    }

    // TROWSUM: row_sum = rowsum(exp_x)
    for (int _row = 0; _row < 8; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 8; _col++) {
            _sum += exp_x[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // FUSED LOOP (2 ops): result=TROWEXPANDDIV(exp_x,row_sum); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v06 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _vb8 = vdupq_n_f32(row_sum[_row][0]);
            float32x4_t _vr7 = vdivq_f32(_v06, _vb8);
            vst1q_f32(&result[_row][_col], _vr7);
            float32x4_t _vs9 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs9);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            result[_row][_col] = exp_x[_row][_col] / row_sum[_row][0];
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}