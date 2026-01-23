// PTO Program: flash_attn_normalize
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_normalize
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 65,792 bytes (64.2 KB)
//   Total capacity (w/ reuse): 65,792 bytes (64.2 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_normalize(float* input_o, float* input_l, float* output) {
    float o_block[64][128];
    float l_vec[64][1];
    float o_final[64][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: o_block = load(input_o[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            o_block[_row][_col] = input_o[_row * 128 + _col];
        }}

    // TLOAD: l_vec = load(input_l[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            l_vec[_row][_col] = input_l[_row * 1 + _col];
        }}

    // TROWEXPANDDIV: o_final = o_block / broadcast(l_vec)
    for (int _row = 0; _row < 64; _row++) {
        float _broadcast_val = l_vec[_row][0];
        for (int _col = 0; _col < 128; _col++) {
            o_final[_row][_col] = o_block[_row][_col] / _broadcast_val;
        }}

    // TSTORE: store(o_final) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = o_final[_row][_col];
        }}

}