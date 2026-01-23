// PTO Program: rmsnorm_tile_256
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile_256
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 658,432 bytes (643.0 KB)
//   Total capacity (w/ reuse): 394,240 bytes (385.0 KB)
//   Reuse savings:            264,192 bytes (40.1%)
//
// ======================================================================

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void rmsnorm_tile_256(float* input, float* weights, float* output, float eps, float inv_cols) {
    float x[256][128];
    float x_sq[256][128];
    float row_sum[256][1];
    float row_mean[256][1];
    float row_rsqrt[256][1];
    float x_norm[256][128];
    float gamma[256][128];
    float result[256][128];

    // Loop fusion: 1 loop overheads saved

    // TLOAD: x = load(input[0, 0])
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x[_row][_col] = input[_row * 128 + _col];
        }}

    // TLOAD: gamma = load(weights[0, 0])
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            gamma[_row][_col] = weights[_row * 128 + _col];
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            x_sq[_row][_col] = x[_row][_col] * x[_row][_col];
        }}
    }

    // TROWSUM: row_sum = rowsum(x_sq)
    for (int _row = 0; _row < 256; _row++) {
        float _sum = 0.0f;
        for (int _col = 0; _col < 128; _col++) {
            _sum += x_sq[_row][_col];
        }
        row_sum[_row][0] = _sum;}

    // LI: Not implemented

    // Fused loop: 1 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            row_mean[_row][_col] = row_sum[_row][_col] * ;
        }}
    }

    // LI: Not implemented

    // Fused loop: 2 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 1; _col++) {
            row_mean[_row][_col] = row_mean[_row][_col] + ;
            row_rsqrt[_row][_col] = 1.0f / sqrtf(row_mean[_row][_col]);
        }}
    }

    // TROWEXPANDMUL: x_norm = x * broadcast(row_rsqrt)
    for (int _row = 0; _row < 256; _row++) {
        float _broadcast_val = row_rsqrt[_row][0];
        for (int _col = 0; _col < 128; _col++) {
            x_norm[_row][_col] = x[_row][_col] * _broadcast_val;
        }}

    // Fused loop: 1 operations
    for (int _row = 0; _row < 256; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            result[_row][_col] = x_norm[_row][_col] * gamma[_row][_col];
        }}
    }

    // TSTORE: store(result) -> output[0, 0]
    for (int _row = 0; _row < 32; _row++) {
        for (int _col = 0; _col < 128; _col++) {
            output[_row * 128 + _col] = result[_row][_col];
        }}

}