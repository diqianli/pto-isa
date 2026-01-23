// PTO Program: rope_tile_128
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rope_tile_128
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 393,216 bytes (384.0 KB)
//   Total capacity (w/ reuse): 262,144 bytes (256.0 KB)
//   Reuse savings:            131,072 bytes (33.3%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rope_tile_128(float* input, float* cos_cache, float* sin_cache, float* output) {
    float x[128][128];
    float cos_pos[128][128];
    float sin_pos[128][128];
    float x_cos[128][128];
    float x_sin[128][128];
    float result[128][128];

    // Loop fusion: 2 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
        }}

    // TLOAD: cos_pos = load(cos_cache[0, 0])
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            cos_pos[_row][_col] = cos_cache[_row * 128 + _col];
        }}

    // TLOAD: sin_pos = load(sin_cache[0, 0])
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            sin_pos[_row][_col] = sin_cache[_row * 128 + _col];
        }}

    // Fused loop: 3 operations
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x_cos[_row][_col] = x[_row][_col] * cos_pos[_row][_col];
            x_sin[_row][_col] = x[_row][_col] * sin_pos[_row][_col];
            result[_row][_col] = x_cos[_row][_col] + x_sin[_row][_col];
        }}
    }

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}