// PTO Program: softmax_tile_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: softmax_tile_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 526,336 bytes (514.0 KB)
//   Total capacity (w/ reuse): 263,168 bytes (257.0 KB)
//   Reuse savings:            263,168 bytes (50.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void softmax_tile_256(float* input, float* output) {
    float x[256][128];
    float row_max[256][1];
    float x_shifted[256][128];
    float exp_x[256][128];
    float row_sum[256][1];
    float result[256][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
        }}

    // TROWMAX: row_max = rowmax(x)
    for (int _row = 0; _row < 256; _row++) {
        float _max = x[_row][0];
        for (int _col = 1; _col < 128; _col++) {
            if (x[_row][_col] > _max) _max = x[_row][_col];
        }
        row_max[_row][0] = _max;}

    // TROWEXPANDSUB: x_shifted = x - broadcast(row_max)
    for (int _row = 0; _row < 256; _row++) {
        float _broadcast_val = row_max[_row][0];
        for (int _col = 0; _col < 128; _col++) {
            x_shifted[_row][_col] = x[_row][_col] - _broadcast_val;
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            exp_x[_row][_col] = expf(x_shifted[_row][_col]);
        }}
    }

    // TROWSUM: row_sum = rowsum(exp_x)
    for (int _row = 0; _row < 256; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 128; _col++) {
            _sum += exp_x[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // TROWEXPANDDIV: result = exp_x / broadcast(row_sum)
    for (int _row = 0; _row < 256; _row++) {
        float _broadcast_val = row_sum[_row][0];
        for (int _col = 0; _col < 128; _col++) {
            result[_row][_col] = exp_x[_row][_col] / _broadcast_val;
        }}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}