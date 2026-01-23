// PTO Program: attention_output_tile_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: attention_output_tile_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 327,680 bytes (320.0 KB)
//   Total capacity (w/ reuse): 327,680 bytes (320.0 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void attention_output_tile_256(float* input_weights, float* input_v, float* output) {
    float weights[256][128];
    float v[128][128];
    float result[256][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: weights = load(input_weights[0, 0])
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            weights[_row][_col] = input_weights[_row * 128 + _col];
        }}

    // TLOAD: v = load(input_v[0, 0])
    for (int _row = 0; _row < 128; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            v[_row][_col] = input_v[_row * 128 + _col];
        }}

    // TMATMUL: result = weights @ v
    for (int _i = 0; _i < 256; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 128; _k++) {
                _sum += weights[_i][_k] * v[_k][_j];}
            result[_i][_j] = _sum;}}

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}