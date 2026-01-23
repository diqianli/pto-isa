// PTO Program: flash_attn_score_block
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_score_block
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     4
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 81,920 bytes (80.0 KB)
//   Reuse savings:            16,384 bytes (16.7%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_score_block(float* input_q, float* input_k, float* output_s, float scale) {
    float q_block[64][128];
    float k_block[64][128];
    float s_block[64][64];
    float s_scaled[64][64];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: q_block = load(input_q[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            q_block[_row][_col] = input_q[_row * 128 + _col];
        }}

    // TLOAD: k_block = load(input_k[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            k_block[_row][_col] = input_k[_row * 128 + _col];
        }}

    // TMATMUL: s_block = q_block @ k_block
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 64; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += q_block[_i][_k] * k_block[_k][_j];}
            s_block[_i][_j] = _sum;}}

    // LI: Not implemented

    // Fused loop: 1 operations
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 64; _col++) {
            s_scaled[_row][_col] = s_block[_row][_col] * ;
        }}
    }

    // TSTORE: store(s_scaled) -> output_s[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_s[_row * 128 + _col] = s_scaled[_row][_col];
        }}

}