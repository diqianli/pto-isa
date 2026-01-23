// PTO Program: flash_attn_softmax_update
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_softmax_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 51,456 bytes (50.2 KB)
//   Total capacity (w/ reuse): 33,792 bytes (33.0 KB)
//   Reuse savings:            17,664 bytes (34.3%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void flash_attn_softmax_update(float* input_s, float* input_m_prev, float* input_l_prev, float* output_m_new, float* output_l_new, float* output_p, float* output_scale_old) {
    float s_block[64][64];
    float m_prev[64][1];
    float l_prev[64][1];
    float m_new[64][1];
    float m_cur[64][1];
    float l_new[64][1];
    float p_block[64][64];
    float s_shifted[64][64];
    float scale_old[64][1];
    float m_diff[64][1];
    float l_scaled[64][1];
    float p_rowsum[64][1];

    // Loop fusion: 2 loop overheads saved

    // TLOAD: s_block = load(input_s[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 64; _col++) {
            s_block[_row][_col] = input_s[_row * 64 + _col];
        }}

    // TLOAD: m_prev = load(input_m_prev[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            m_prev[_row][_col] = input_m_prev[_row * 1 + _col];
        }}

    // TLOAD: l_prev = load(input_l_prev[0, 0])
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            l_prev[_row][_col] = input_l_prev[_row * 1 + _col];
        }}

    // TROWMAX: m_cur = rowmax(s_block)
    for (int _row = 0; _row < 64; _row++) {
        float _max = s_block[_row][0];
        for (int _col = 1; _col < 64; _col++) {
            if (s_block[_row][_col] > _max) _max = s_block[_row][_col];
        }
        m_cur[_row][0] = _max;}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            m_new[_row][_col] = (m_prev[_row][_col] > m_cur[_row][_col]) ? m_prev[_row][_col] : m_cur[_row][_col];
        }}
    }

    // TROWEXPANDSUB: s_shifted = s_block - broadcast(m_new)
    for (int _row = 0; _row < 64; _row++) {
        float _broadcast_val = m_new[_row][0];
        for (int _col = 0; _col < 64; _col++) {
            s_shifted[_row][_col] = s_block[_row][_col] - _broadcast_val;
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 64; _col++) {
            p_block[_row][_col] = expf(s_shifted[_row][_col]);
        }}
    }

    // Fused loop: 3 operations
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            m_diff[_row][_col] = m_prev[_row][_col] - m_new[_row][_col];
            scale_old[_row][_col] = expf(m_diff[_row][_col]);
            l_scaled[_row][_col] = scale_old[_row][_col] * l_prev[_row][_col];
        }}
    }

    // TROWSUM: p_rowsum = rowsum(p_block)
    for (int _row = 0; _row < 64; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 64; _col++) {
            _sum += p_block[_row][_col];
        }
        p_rowsum[_row][0] = _sum;}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 64; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            l_new[_row][_col] = l_scaled[_row][_col] + p_rowsum[_row][_col];
        }}
    }

    // TSTORE: store(m_new) -> output_m_new[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_m_new[_row * 128 + _col] = m_new[_row][_col];
        }}

    // TSTORE: store(l_new) -> output_l_new[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_l_new[_row * 128 + _col] = l_new[_row][_col];
        }}

    // TSTORE: store(p_block) -> output_p[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_p[_row * 128 + _col] = p_block[_row][_col];
        }}

    // TSTORE: store(scale_old) -> output_scale_old[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output_scale_old[_row * 128 + _col] = scale_old[_row][_col];
        }}

}