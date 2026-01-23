// PTO Program: tile_muls_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_muls_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     2
//   Total capacity (no reuse): 262,144 bytes (256.0 KB)
//   Total capacity (w/ reuse): 262,144 bytes (256.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_muls_256(float* input, float* output, float scale) {
    float a[256][128];
    float result[256][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: a = load(input[0, 0])
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            a[_row][_col] = input[_row * 128 + _col];
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            result[_row][_col] = a[_row][_col] * ;
        }}
    }

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}