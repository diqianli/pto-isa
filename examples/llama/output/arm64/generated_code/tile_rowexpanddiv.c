// PTO Program: tile_rowexpanddiv
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: tile_rowexpanddiv
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 32,896 bytes (32.1 KB)
//   Total capacity (w/ reuse): 32,896 bytes (32.1 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tile_rowexpanddiv(float* input_x, float* input_row, float* output) {
    float x[32][128];
    float row_vals[32][1];
    float result[32][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input_x[0, 0])
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x[_row][_col] = input_x[_row * 128 + _col];
        }}

    // TLOAD: row_vals = load(input_row[0, 0])
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            row_vals[_row][_col] = input_row[_row * 1 + _col];
        }}

    // TROWEXPANDDIV: result = x / broadcast(row_vals)
    for (int _row = 0; _row < 32; _row++) {
        float _broadcast_val = row_vals[_row][0];
        for (int _col = 0; _col < 128; _col++) {
            result[_row][_col] = x[_row][_col] / _broadcast_val;
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}