// PTO Program: flash_attn_output_update
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_output_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     7
//   Total capacity (no reuse): 180,480 bytes (176.2 KB)
//   Total capacity (w/ reuse): 114,944 bytes (112.2 KB)
//   Reuse savings:            65,536 bytes (36.3%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_output_update(float* input_o_prev, float* input_p, float* input_v, float* input_scale, float* output_o) {
    float o_prev[64][128];
    float p_block[64][64];
    float v_block[64][128];
    float scale_old[64][1];
    float o_scaled[64][128];
    float pv[64][128];
    float o_new[64][128];

    // Loop fusion: 0 loop overheads saved

    // TLOAD: o_prev = load(input_o_prev[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            o_prev[_row][_col] = input_o_prev[_row * 128 + _col];
        }}

    // TLOAD: p_block = load(input_p[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 64; _col++) {
            p_block[_row][_col] = input_p[_row * 64 + _col];
        }}

    // TLOAD: v_block = load(input_v[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            v_block[_row][_col] = input_v[_row * 128 + _col];
        }}

    // TLOAD: scale_old = load(input_scale[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            scale_old[_row][_col] = input_scale[_row * 1 + _col];
        }}

    // TROWEXPANDMUL: o_scaled = o_prev * broadcast(scale_old)
    for (int _row = 0; _row < 64; _row++) {
        float _broadcast_val = scale_old[_row][0];
        for (int _col = 0; _col < 128; _col++) {
            o_scaled[_row][_col] = o_prev[_row][_col] * _broadcast_val;
        }}

    // TMATMUL: pv = p_block @ v_block
    for (int _i = 0; _i < 64; _i++) {
        for (int _j = 0; _j < 128; _j++) {
            float _sum = 0.0f;
            for (int _k = 0; _k < 64; _k++) {
                _sum += p_block[_i][_k] * v_block[_k][_j];}
            pv[_i][_j] = _sum;}}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            o_new[_row][_col] = o_scaled[_row][_col] + pv[_row][_col];
        }}
    }

    // TSTORE: store(o_new) -> output_o[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_o[_row * 128 + _col] = o_new[_row][_col];
        }}

}