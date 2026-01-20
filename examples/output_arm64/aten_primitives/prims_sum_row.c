// PTO Program: prims_sum_row
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void prims_sum_row(float* input, float* output) {
    float x[1][4096];
    float result[1][1];

    // Loop fusion: 0 loop overheads saved

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input[_row * 4096 + _col];
        }
    }

    // TROWSUM: result = rowsum(x)
    for (int _row = 0; _row < 1; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 4096; _col++) {
            _sum += x[_row][_col];
        }
        result[_row][0] = _sum;}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs1 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs1);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

    // FUSED LOOP (1 ops): x=TLOAD(input,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl2);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input[_row * 4096 + _col];
        }
    }

    // TROWSUM: result = rowsum(x)
    for (int _row = 0; _row < 1; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 4096; _col++) {
            _sum += x[_row][_col];
        }
        result[_row][0] = _sum;}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 1; _col += 4) {
            float32x4_t _vs3 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 1 + _col], _vs3);
        }
        // Scalar cleanup
        for (; _col < 1; _col++) {
            output[_row * 1 + _col] = result[_row][_col];
        }
    }

}