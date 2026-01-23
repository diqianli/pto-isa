// PTO Program: swiglu_tile
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            65,536 bytes (50.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void swiglu_tile(float* input_gate, float* input_up, float* output) {
    float gate[32][128];
    float up[32][128];
    float neg_gate[32][128];
    float exp_neg_gate[32][128];
    float one_plus_exp[32][128];
    float sigmoid_gate[32][128];
    float gate_silu[32][128];
    float result[32][128];

    // Loop fusion: 5 loop overheads saved

    // TLOAD: gate = load(input_gate[0, 0])
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            gate[_row][_col] = input_gate[_row * 128 + _col];
        }}

    // TLOAD: up = load(input_up[0, 0])
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            up[_row][_col] = input_up[_row * 128 + _col];
        }}

    // Fused loop: 6 operations
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            neg_gate[_row][_col] = -gate[_row][_col];
            exp_neg_gate[_row][_col] = expf(neg_gate[_row][_col]);
            one_plus_exp[_row][_col] = exp_neg_gate[_row][_col] + ;
            sigmoid_gate[_row][_col] = 1.0f / one_plus_exp[_row][_col];
            gate_silu[_row][_col] = gate[_row][_col] * sigmoid_gate[_row][_col];
            result[_row][_col] = gate_silu[_row][_col] * up[_row][_col];
        }}
    }

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}