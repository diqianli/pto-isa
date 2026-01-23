// PTO Program: flash_attn_init_state
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_init_state
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     3
//   Total capacity (no reuse): 33,280 bytes (32.5 KB)
//   Total capacity (w/ reuse): 33,280 bytes (32.5 KB)
//   Reuse savings:            0 bytes (0.0%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_init_state(float* input_zeros_large, float* input_zeros_small, float* input_neg_inf, float* output_o, float* output_l, float* output_m) {
    float o_init[64][128];
    float l_init[64][1];
    float m_init[64][1];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: o_init = load(input_zeros_large[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            o_init[_row][_col] = input_zeros_large[_row * 128 + _col];
        }}

    // TLOAD: l_init = load(input_zeros_small[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            l_init[_row][_col] = input_zeros_small[_row * 1 + _col];
        }}

    // TLOAD: m_init = load(input_neg_inf[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            m_init[_row][_col] = input_neg_inf[_row * 1 + _col];
        }}

    // TSTORE: store(o_init) -> output_o[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_o[_row * 128 + _col] = o_init[_row][_col];
        }}

    // TSTORE: store(l_init) -> output_l[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_l[_row * 128 + _col] = l_init[_row][_col];
        }}

    // TSTORE: store(m_init) -> output_m[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_m[_row * 128 + _col] = m_init[_row][_col];
        }}

}