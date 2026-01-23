// PTO Program: attention_score_tile_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: attention_score_tile_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 458,752 bytes (448.0 KB)
//   Total capacity (w/ reuse): 327,680 bytes (320.0 KB)
//   Reuse savings:            131,072 bytes (28.6%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void attention_score_tile_256(float* input_q, float* input_kt, float* output, float scale) {
    float q[256][128];
    float k_t[128][128];
    float scores[256][128];
    float scaled_scores[256][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: q = load(input_q[0, 0])
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            q[_row][_col] = input_q[_row * 128 + _col];
        }}

    // TLOAD: k_t = load(input_kt[0, 0])
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            k_t[_row][_col] = input_kt[_row * 128 + _col];
        }}

    // TMATMUL: scores = q @ k_t
    for (int _i = 0; _i < 256; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += q[_i][_k] * k_t[_k][_j];}
            scores[_i][_j] = _sum;}}

    // LI: Not implemented

    // Fused loop: 1 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            scaled_scores[_row][_col] = scores[_row][_col] * ;
        }}
    }

    // TSTORE: store(scaled_scores) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = scaled_scores[_row][_col];
        }}

}