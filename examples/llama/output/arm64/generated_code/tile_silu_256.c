// PTO Program: tile_silu_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_silu_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 786,432 bytes (768.0 KB)
//   Total capacity (w/ reuse): 393,216 bytes (384.0 KB)
//   Reuse savings:            393,216 bytes (50.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_silu_256(float* input, float* output) {
    float x[256][128];
    float neg_x[256][128];
    float exp_neg_x[256][128];
    float one_plus_exp[256][128];
    float sigmoid[256][128];
    float result[256][128];

    // Loop fusion: 4 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
        }}

    // Fused loop: 5 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            one_plus_exp[_row][_col] = exp_neg_x[_row][_col] + ;
            sigmoid[_row][_col] = 1.0f / one_plus_exp[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid[_row][_col];
        }}
    }

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}