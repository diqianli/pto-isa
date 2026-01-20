// PTO Program: attention_score_tile
// Function Type: InCore (tile-level computation)
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void attention_score_tile(float* input_q, float* input_kt, float* output) {
    float q[8][8];
    float k_t[8][8];
    float scores[8][8];
    float scaled_scores[8][8];

    // Loop fusion: 2 loop overheads saved

    // FUSED LOOP (2 ops): q=TLOAD(input_q,0,0); k_t=TLOAD(input_kt,0,0)
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input_q[_row * 8 + _col]);
            vst1q_f32(&q[_row][_col], _vl0);
            float32x4_t _vl1 = vld1q_f32(&input_kt[_row * 8 + _col]);
            vst1q_f32(&k_t[_row][_col], _vl1);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            q[_row][_col] = input_q[_row * 8 + _col];
            k_t[_row][_col] = input_kt[_row * 8 + _col];
        }
    }

    // TMATMUL: scores = q @ k_t
    for (int _i = 0; _i < 8; _i++) {
        for (int _j = 0; _j < 8; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 8; _k++) {
                _sum += q[_i][_k] * k_t[_k][_j];}
            scores[_i][_j] = _sum;}}

    int scale = 0.08838834764831843;

    // FUSED LOOP (2 ops): scaled_scores=TMULS(scores,scalef); output=TSTORE(scaled_scores,0,0)
    float32x4_t _vs2 = vdupq_n_f32(scalef);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v3 = vld1q_f32(&scores[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v3, _vs2);
            vst1q_f32(&scaled_scores[_row][_col], _vr4);
            float32x4_t _vs5 = vld1q_f32(&scaled_scores[_row][_col]);
            vst1q_f32(&output[_row * 8 + _col], _vs5);
        }
        // Scalar cleanup
        for (; _col < 8; _col++) {
            scaled_scores[_row][_col] = scores[_row][_col] * scalef;
            output[_row * 8 + _col] = scaled_scores[_row][_col];
        }
    }

}