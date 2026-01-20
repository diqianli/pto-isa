// PTO Program: attention_output_tile
// Function Type: InCore (tile-level computation)
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void attention_output_tile(float* input_weights, float* input_v, float* output) {
    float weights[8][8];
    float v[8][8];
    float result[8][8];

    // Loop fusion: 1 loop overheads saved

    // FUSED LOOP (2 ops): weights=TLOAD(input_weights,0,0); v=TLOAD(input_v,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_weights[_row * 8 + _col]);
            vst1q_f32(&weights[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_v[_row * 8 + _col]);
            vst1q_f32(&v[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            weights[_row][_col] = input_weights[_row * 8 + _col];
            v[_row][_col] = input_v[_row * 8 + _col];
        }
    }

    // TMATMUL: result = weights @ v
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += weights[_i][_k] * v[_k][_j];}
            result[_i][_j] = _sum;}}

    // FUSED LOOP (1 ops): output=TSTORE(result,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vs2 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs2);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            output[_row * 8 + _col] = result[_row][_col];
        }
    }

}