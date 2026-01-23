// PTO Program: llama_layer_dynamic
// Function Type: Orchestration (control flow only)
// Orchestration function - builds task graph using PTO runtime
#include "pto_runtime.h"
#include "pto_runtime.c"  // Include for standalone build
#include <string.h>  // For strcmp in main
#include <time.h>    // For benchmark timing

// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void llama_layer_dynamic(PTORuntime* rt, float* input, float* output, float* attn_norm_weights, float* wq, float* wk, float* wv, float* wo, float* cos_cache, float* sin_cache, float* mlp_norm_weights, float* w_gate, float* w_up, float* w_down, float* all_q_tiles, float* all_k_tiles, float* all_v_tiles, float* all_q_rope, float* all_k_rope, float* all_attn_out, float* all_m_vec, float* all_l_vec, float* all_hidden, float* temp_norm, float* temp_scores, float* temp_attn_weights, float* temp_scale, float* temp_gate, float* temp_up, float* temp_swiglu, float* temp_mlp_out, float* const_zeros_large, float* const_zeros_small, float* const_neg_inf, int32_t seq_len, int32_t tile_rows, int32_t num_tiles, int32_t zero) {

    // Loop fusion: 0 loop overheads saved

    // LI: Not implemented

    // LI: Not implemented

    // Binary-expanded loop: tile_i in [0, num_tiles), max_range=4096
    int tile_i_remaining_0 = num_tiles - 0;
    int tile_i_base_0 = 0;
    if (tile_i_remaining_0 >= 4096) {
        for (int tile_i = tile_i_base_0; tile_i < tile_i_base_0 + 4096; tile_i += 1) {
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t0, input, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t0, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t1, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t1, wq, 0, 0, 32, 128);
            pto_task_add_output(rt, t1, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t2, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t2, wk, 0, 0, 32, 128);
            pto_task_add_output(rt, t2, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t3, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t3, wv, 0, 0, 32, 128);
            pto_task_add_output(rt, t3, all_v_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t4, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t4, all_q_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t5, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t5, all_k_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
        }
        tile_i_base_0 += 4096;
        tile_i_remaining_0 -= 4096;
    }
    if (tile_i_remaining_0 >= 2048) {
        for (int tile_i = tile_i_base_0; tile_i < tile_i_base_0 + 2048; tile_i += 1) {
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t0, input, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t0, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t1, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t1, wq, 0, 0, 32, 128);
            pto_task_add_output(rt, t1, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t2, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t2, wk, 0, 0, 32, 128);
            pto_task_add_output(rt, t2, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t3, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t3, wv, 0, 0, 32, 128);
            pto_task_add_output(rt, t3, all_v_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t4, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t4, all_q_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t5, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t5, all_k_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
        }
        tile_i_base_0 += 2048;
        tile_i_remaining_0 -= 2048;
    }
    if (tile_i_remaining_0 >= 1024) {
        for (int tile_i = tile_i_base_0; tile_i < tile_i_base_0 + 1024; tile_i += 1) {
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t0, input, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t0, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t1, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t1, wq, 0, 0, 32, 128);
            pto_task_add_output(rt, t1, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t2, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t2, wk, 0, 0, 32, 128);
            pto_task_add_output(rt, t2, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t3, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t3, wv, 0, 0, 32, 128);
            pto_task_add_output(rt, t3, all_v_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t4, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t4, all_q_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t5, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t5, all_k_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
        }
        tile_i_base_0 += 1024;
        tile_i_remaining_0 -= 1024;
    }
    if (tile_i_remaining_0 >= 512) {
        for (int tile_i = tile_i_base_0; tile_i < tile_i_base_0 + 512; tile_i += 1) {
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t0, input, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t0, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t1, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t1, wq, 0, 0, 32, 128);
            pto_task_add_output(rt, t1, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t2, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t2, wk, 0, 0, 32, 128);
            pto_task_add_output(rt, t2, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t3, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t3, wv, 0, 0, 32, 128);
            pto_task_add_output(rt, t3, all_v_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t4, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t4, all_q_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t5, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t5, all_k_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
        }
        tile_i_base_0 += 512;
        tile_i_remaining_0 -= 512;
    }
    if (tile_i_remaining_0 >= 256) {
        for (int tile_i = tile_i_base_0; tile_i < tile_i_base_0 + 256; tile_i += 1) {
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t0, input, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t0, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t1, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t1, wq, 0, 0, 32, 128);
            pto_task_add_output(rt, t1, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t2, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t2, wk, 0, 0, 32, 128);
            pto_task_add_output(rt, t2, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t3, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t3, wv, 0, 0, 32, 128);
            pto_task_add_output(rt, t3, all_v_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t4, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t4, all_q_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t5, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t5, all_k_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
        }
        tile_i_base_0 += 256;
        tile_i_remaining_0 -= 256;
    }
    // Residual loop for remaining < 256
    for (int tile_i = tile_i_base_0; tile_i < tile_i_base_0 + tile_i_remaining_0; tile_i += 1) {
    
            // Task 0: rmsnorm_tile
            int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t0, input, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t0, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t0);
    
    
            // Task 1: tile_matmul
            int32_t t1 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t1, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t1, wq, 0, 0, 32, 128);
            pto_task_add_output(rt, t1, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t1);
    
    
            // Task 2: tile_matmul
            int32_t t2 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t2, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t2, wk, 0, 0, 32, 128);
            pto_task_add_output(rt, t2, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t2);
    
    
            // Task 3: tile_matmul
            int32_t t3 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t3, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t3, wv, 0, 0, 32, 128);
            pto_task_add_output(rt, t3, all_v_tiles, tile_i, 0, 32, 128);
            pto_task_submit(rt, t3);
    
    
            // Task 4: rope_tile
            int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t4, all_q_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t4, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t4, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t4, all_q_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t4);
    
    
            // Task 5: rope_tile
            int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536, 0);
            pto_task_add_input(rt, t5, all_k_tiles, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t5, cos_cache, 0, 0, 32, 128);
            pto_task_add_input(rt, t5, sin_cache, 0, 0, 32, 128);
            pto_task_add_output(rt, t5, all_k_rope, tile_i, 0, 32, 128);
            pto_task_submit(rt, t5);
    
    
    }

    // Binary-expanded loop: q_tile in [0, num_tiles), max_range=4096
    int q_tile_remaining_1 = num_tiles - 0;
    int q_tile_base_1 = 0;
    if (q_tile_remaining_1 >= 4096) {
        for (int q_tile = q_tile_base_1; q_tile < q_tile_base_1 + 4096; q_tile += 1) {
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280, 0);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 32, 128);
            pto_task_add_output(rt, t6, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_m_vec, q_tile, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // @BINARY_EXPAND: max_range=4096, min_range=256, bits=[4096,2048,1024,512,256] tile_levels={4096:256,2048:128,1024:64,512:64,256:64,0:32}
            for (int kv_tile = 0; kv_tile < num_tiles; kv_tile += 1) {
    
                // Task 7: flash_attn_score_block
                int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 81920, 1);
                pto_task_add_input(rt, t7, all_q_rope, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t7, all_k_rope, kv_tile, 0, 32, 128);
                pto_task_add_output(rt, t7, temp_scores, q_tile, 0, 32, 128);
                pto_task_submit(rt, t7);
    
    
                // Task 8: flash_attn_softmax_update
                int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 33792, 0);
                pto_task_add_input(rt, t8, temp_scores, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_scale, q_tile, 0, 32, 128);
                pto_task_submit(rt, t8);
    
    
                // Task 9: flash_attn_output_update
                int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 114944, 1);
                pto_task_add_input(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, all_v_tiles, kv_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_scale, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_submit(rt, t9);
    
    
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792, 0);
            pto_task_add_input(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_input(rt, t10, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
        }
        q_tile_base_1 += 4096;
        q_tile_remaining_1 -= 4096;
    }
    if (q_tile_remaining_1 >= 2048) {
        for (int q_tile = q_tile_base_1; q_tile < q_tile_base_1 + 2048; q_tile += 1) {
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280, 0);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 32, 128);
            pto_task_add_output(rt, t6, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_m_vec, q_tile, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // @BINARY_EXPAND: max_range=4096, min_range=256, bits=[4096,2048,1024,512,256] tile_levels={4096:256,2048:128,1024:64,512:64,256:64,0:32}
            for (int kv_tile = 0; kv_tile < num_tiles; kv_tile += 1) {
    
                // Task 7: flash_attn_score_block
                int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 81920, 1);
                pto_task_add_input(rt, t7, all_q_rope, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t7, all_k_rope, kv_tile, 0, 32, 128);
                pto_task_add_output(rt, t7, temp_scores, q_tile, 0, 32, 128);
                pto_task_submit(rt, t7);
    
    
                // Task 8: flash_attn_softmax_update
                int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 33792, 0);
                pto_task_add_input(rt, t8, temp_scores, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_scale, q_tile, 0, 32, 128);
                pto_task_submit(rt, t8);
    
    
                // Task 9: flash_attn_output_update
                int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 114944, 1);
                pto_task_add_input(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, all_v_tiles, kv_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_scale, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_submit(rt, t9);
    
    
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792, 0);
            pto_task_add_input(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_input(rt, t10, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
        }
        q_tile_base_1 += 2048;
        q_tile_remaining_1 -= 2048;
    }
    if (q_tile_remaining_1 >= 1024) {
        for (int q_tile = q_tile_base_1; q_tile < q_tile_base_1 + 1024; q_tile += 1) {
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280, 0);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 32, 128);
            pto_task_add_output(rt, t6, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_m_vec, q_tile, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // @BINARY_EXPAND: max_range=4096, min_range=256, bits=[4096,2048,1024,512,256] tile_levels={4096:256,2048:128,1024:64,512:64,256:64,0:32}
            for (int kv_tile = 0; kv_tile < num_tiles; kv_tile += 1) {
    
                // Task 7: flash_attn_score_block
                int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 81920, 1);
                pto_task_add_input(rt, t7, all_q_rope, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t7, all_k_rope, kv_tile, 0, 32, 128);
                pto_task_add_output(rt, t7, temp_scores, q_tile, 0, 32, 128);
                pto_task_submit(rt, t7);
    
    
                // Task 8: flash_attn_softmax_update
                int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 33792, 0);
                pto_task_add_input(rt, t8, temp_scores, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_scale, q_tile, 0, 32, 128);
                pto_task_submit(rt, t8);
    
    
                // Task 9: flash_attn_output_update
                int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 114944, 1);
                pto_task_add_input(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, all_v_tiles, kv_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_scale, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_submit(rt, t9);
    
    
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792, 0);
            pto_task_add_input(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_input(rt, t10, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
        }
        q_tile_base_1 += 1024;
        q_tile_remaining_1 -= 1024;
    }
    if (q_tile_remaining_1 >= 512) {
        for (int q_tile = q_tile_base_1; q_tile < q_tile_base_1 + 512; q_tile += 1) {
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280, 0);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 32, 128);
            pto_task_add_output(rt, t6, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_m_vec, q_tile, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // @BINARY_EXPAND: max_range=4096, min_range=256, bits=[4096,2048,1024,512,256] tile_levels={4096:256,2048:128,1024:64,512:64,256:64,0:32}
            for (int kv_tile = 0; kv_tile < num_tiles; kv_tile += 1) {
    
                // Task 7: flash_attn_score_block
                int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 81920, 1);
                pto_task_add_input(rt, t7, all_q_rope, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t7, all_k_rope, kv_tile, 0, 32, 128);
                pto_task_add_output(rt, t7, temp_scores, q_tile, 0, 32, 128);
                pto_task_submit(rt, t7);
    
    
                // Task 8: flash_attn_softmax_update
                int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 33792, 0);
                pto_task_add_input(rt, t8, temp_scores, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_scale, q_tile, 0, 32, 128);
                pto_task_submit(rt, t8);
    
    
                // Task 9: flash_attn_output_update
                int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 114944, 1);
                pto_task_add_input(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, all_v_tiles, kv_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_scale, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_submit(rt, t9);
    
    
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792, 0);
            pto_task_add_input(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_input(rt, t10, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
        }
        q_tile_base_1 += 512;
        q_tile_remaining_1 -= 512;
    }
    if (q_tile_remaining_1 >= 256) {
        for (int q_tile = q_tile_base_1; q_tile < q_tile_base_1 + 256; q_tile += 1) {
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280, 0);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 32, 128);
            pto_task_add_output(rt, t6, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_m_vec, q_tile, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // @BINARY_EXPAND: max_range=4096, min_range=256, bits=[4096,2048,1024,512,256] tile_levels={4096:256,2048:128,1024:64,512:64,256:64,0:32}
            for (int kv_tile = 0; kv_tile < num_tiles; kv_tile += 1) {
    
                // Task 7: flash_attn_score_block
                int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 81920, 1);
                pto_task_add_input(rt, t7, all_q_rope, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t7, all_k_rope, kv_tile, 0, 32, 128);
                pto_task_add_output(rt, t7, temp_scores, q_tile, 0, 32, 128);
                pto_task_submit(rt, t7);
    
    
                // Task 8: flash_attn_softmax_update
                int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 33792, 0);
                pto_task_add_input(rt, t8, temp_scores, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_scale, q_tile, 0, 32, 128);
                pto_task_submit(rt, t8);
    
    
                // Task 9: flash_attn_output_update
                int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 114944, 1);
                pto_task_add_input(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, all_v_tiles, kv_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_scale, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_submit(rt, t9);
    
    
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792, 0);
            pto_task_add_input(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_input(rt, t10, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
        }
        q_tile_base_1 += 256;
        q_tile_remaining_1 -= 256;
    }
    // Residual loop for remaining < 256
    for (int q_tile = q_tile_base_1; q_tile < q_tile_base_1 + q_tile_remaining_1; q_tile += 1) {
    
            // Task 6: flash_attn_init_state
            int32_t t6 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280, 0);
            pto_task_add_input(rt, t6, const_zeros_large, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_zeros_small, 0, 0, 32, 128);
            pto_task_add_input(rt, t6, const_neg_inf, 0, 0, 32, 128);
            pto_task_add_output(rt, t6, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t6, all_m_vec, q_tile, 0, 32, 128);
            pto_task_submit(rt, t6);
    
    
            // @BINARY_EXPAND: max_range=4096, min_range=256, bits=[4096,2048,1024,512,256] tile_levels={4096:256,2048:128,1024:64,512:64,256:64,0:32}
            for (int kv_tile = 0; kv_tile < num_tiles; kv_tile += 1) {
    
                // Task 7: flash_attn_score_block
                int32_t t7 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 81920, 1);
                pto_task_add_input(rt, t7, all_q_rope, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t7, all_k_rope, kv_tile, 0, 32, 128);
                pto_task_add_output(rt, t7, temp_scores, q_tile, 0, 32, 128);
                pto_task_submit(rt, t7);
    
    
                // Task 8: flash_attn_softmax_update
                int32_t t8 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 33792, 0);
                pto_task_add_input(rt, t8, temp_scores, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_m_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, all_l_vec, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t8, temp_scale, q_tile, 0, 32, 128);
                pto_task_submit(rt, t8);
    
    
                // Task 9: flash_attn_output_update
                int32_t t9 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 114944, 1);
                pto_task_add_input(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_attn_weights, q_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, all_v_tiles, kv_tile, 0, 32, 128);
                pto_task_add_input(rt, t9, temp_scale, q_tile, 0, 32, 128);
                pto_task_add_output(rt, t9, all_attn_out, q_tile, 0, 32, 128);
                pto_task_submit(rt, t9);
    
    
            }
    
            // Task 10: flash_attn_normalize
            int32_t t10 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792, 0);
            pto_task_add_input(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_add_input(rt, t10, all_l_vec, q_tile, 0, 32, 128);
            pto_task_add_output(rt, t10, all_attn_out, q_tile, 0, 32, 128);
            pto_task_submit(rt, t10);
    
    
    }

    // Binary-expanded loop: tile_i in [0, num_tiles), max_range=4096
    int tile_i_remaining_2 = num_tiles - 0;
    int tile_i_base_2 = 0;
    if (tile_i_remaining_2 >= 4096) {
        for (int tile_i = tile_i_base_2; tile_i < tile_i_base_2 + 4096; tile_i += 1) {
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t11, all_attn_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t11, wo, 0, 0, 32, 128);
            pto_task_add_output(rt, t11, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t12, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t12, input, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t12, all_hidden, tile_i, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t13, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t13, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t14, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 32, 128);
            pto_task_add_output(rt, t14, temp_gate, tile_i, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t15, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t15, w_up, 0, 0, 32, 128);
            pto_task_add_output(rt, t15, temp_up, tile_i, 0, 32, 128);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536, 0);
            pto_task_add_input(rt, t16, temp_gate, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t16, temp_up, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t16, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t17, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t17, w_down, 0, 0, 32, 128);
            pto_task_add_output(rt, t17, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t18, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t18, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t18, output, tile_i, 0, 32, 128);
            pto_task_submit(rt, t18);
    
    
        }
        tile_i_base_2 += 4096;
        tile_i_remaining_2 -= 4096;
    }
    if (tile_i_remaining_2 >= 2048) {
        for (int tile_i = tile_i_base_2; tile_i < tile_i_base_2 + 2048; tile_i += 1) {
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t11, all_attn_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t11, wo, 0, 0, 32, 128);
            pto_task_add_output(rt, t11, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t12, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t12, input, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t12, all_hidden, tile_i, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t13, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t13, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t14, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 32, 128);
            pto_task_add_output(rt, t14, temp_gate, tile_i, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t15, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t15, w_up, 0, 0, 32, 128);
            pto_task_add_output(rt, t15, temp_up, tile_i, 0, 32, 128);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536, 0);
            pto_task_add_input(rt, t16, temp_gate, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t16, temp_up, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t16, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t17, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t17, w_down, 0, 0, 32, 128);
            pto_task_add_output(rt, t17, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t18, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t18, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t18, output, tile_i, 0, 32, 128);
            pto_task_submit(rt, t18);
    
    
        }
        tile_i_base_2 += 2048;
        tile_i_remaining_2 -= 2048;
    }
    if (tile_i_remaining_2 >= 1024) {
        for (int tile_i = tile_i_base_2; tile_i < tile_i_base_2 + 1024; tile_i += 1) {
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t11, all_attn_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t11, wo, 0, 0, 32, 128);
            pto_task_add_output(rt, t11, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t12, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t12, input, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t12, all_hidden, tile_i, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t13, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t13, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t14, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 32, 128);
            pto_task_add_output(rt, t14, temp_gate, tile_i, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t15, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t15, w_up, 0, 0, 32, 128);
            pto_task_add_output(rt, t15, temp_up, tile_i, 0, 32, 128);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536, 0);
            pto_task_add_input(rt, t16, temp_gate, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t16, temp_up, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t16, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t17, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t17, w_down, 0, 0, 32, 128);
            pto_task_add_output(rt, t17, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t18, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t18, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t18, output, tile_i, 0, 32, 128);
            pto_task_submit(rt, t18);
    
    
        }
        tile_i_base_2 += 1024;
        tile_i_remaining_2 -= 1024;
    }
    if (tile_i_remaining_2 >= 512) {
        for (int tile_i = tile_i_base_2; tile_i < tile_i_base_2 + 512; tile_i += 1) {
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t11, all_attn_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t11, wo, 0, 0, 32, 128);
            pto_task_add_output(rt, t11, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t12, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t12, input, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t12, all_hidden, tile_i, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t13, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t13, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t14, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 32, 128);
            pto_task_add_output(rt, t14, temp_gate, tile_i, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t15, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t15, w_up, 0, 0, 32, 128);
            pto_task_add_output(rt, t15, temp_up, tile_i, 0, 32, 128);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536, 0);
            pto_task_add_input(rt, t16, temp_gate, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t16, temp_up, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t16, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t17, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t17, w_down, 0, 0, 32, 128);
            pto_task_add_output(rt, t17, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t18, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t18, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t18, output, tile_i, 0, 32, 128);
            pto_task_submit(rt, t18);
    
    
        }
        tile_i_base_2 += 512;
        tile_i_remaining_2 -= 512;
    }
    if (tile_i_remaining_2 >= 256) {
        for (int tile_i = tile_i_base_2; tile_i < tile_i_base_2 + 256; tile_i += 1) {
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t11, all_attn_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t11, wo, 0, 0, 32, 128);
            pto_task_add_output(rt, t11, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t12, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t12, input, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t12, all_hidden, tile_i, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t13, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t13, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t14, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 32, 128);
            pto_task_add_output(rt, t14, temp_gate, tile_i, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t15, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t15, w_up, 0, 0, 32, 128);
            pto_task_add_output(rt, t15, temp_up, tile_i, 0, 32, 128);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536, 0);
            pto_task_add_input(rt, t16, temp_gate, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t16, temp_up, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t16, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t17, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t17, w_down, 0, 0, 32, 128);
            pto_task_add_output(rt, t17, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t18, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t18, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t18, output, tile_i, 0, 32, 128);
            pto_task_submit(rt, t18);
    
    
        }
        tile_i_base_2 += 256;
        tile_i_remaining_2 -= 256;
    }
    // Residual loop for remaining < 256
    for (int tile_i = tile_i_base_2; tile_i < tile_i_base_2 + tile_i_remaining_2; tile_i += 1) {
    
            // Task 11: tile_matmul
            int32_t t11 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t11, all_attn_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t11, wo, 0, 0, 32, 128);
            pto_task_add_output(rt, t11, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t11);
    
    
            // Task 12: residual_add_tile
            int32_t t12 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t12, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t12, input, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t12, all_hidden, tile_i, 0, 32, 128);
            pto_task_submit(rt, t12);
    
    
            // Task 13: rmsnorm_tile
            int32_t t13 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49280, 0);
            pto_task_add_input(rt, t13, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t13, mlp_norm_weights, 0, 0, 32, 128);
            pto_task_add_output(rt, t13, temp_norm, tile_i, 0, 32, 128);
            pto_task_submit(rt, t13);
    
    
            // Task 14: tile_matmul
            int32_t t14 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t14, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t14, w_gate, 0, 0, 32, 128);
            pto_task_add_output(rt, t14, temp_gate, tile_i, 0, 32, 128);
            pto_task_submit(rt, t14);
    
    
            // Task 15: tile_matmul
            int32_t t15 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t15, temp_norm, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t15, w_up, 0, 0, 32, 128);
            pto_task_add_output(rt, t15, temp_up, tile_i, 0, 32, 128);
            pto_task_submit(rt, t15);
    
    
            // Task 16: swiglu_tile
            int32_t t16 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536, 0);
            pto_task_add_input(rt, t16, temp_gate, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t16, temp_up, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t16, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_submit(rt, t16);
    
    
            // Task 17: tile_matmul
            int32_t t17 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304, 1);
            pto_task_add_input(rt, t17, temp_swiglu, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t17, w_down, 0, 0, 32, 128);
            pto_task_add_output(rt, t17, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_submit(rt, t17);
    
    
            // Task 18: residual_add_tile
            int32_t t18 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152, 0);
            pto_task_add_input(rt, t18, temp_mlp_out, tile_i, 0, 32, 128);
            pto_task_add_input(rt, t18, all_hidden, tile_i, 0, 32, 128);
            pto_task_add_output(rt, t18, output, tile_i, 0, 32, 128);
            pto_task_submit(rt, t18);
    
    
    }

}
// =============================================================================
// Main Function for ARM64 Standalone Execution
// =============================================================================
// Usage: llama_layer_dynamic [--benchmark-only] [seq_len] [tile_rows] [num_tiles] [zero]
// Flags:
//   --benchmark-only  - Only run orchestration (skip execution), output stats

int main(int argc, char** argv) {
    // Check for --benchmark-only flag
    int benchmark_only = 0;
    int arg_offset = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark-only") == 0) {
            benchmark_only = 1;
            arg_offset = 1;
            break;
        }
    }
    
    printf("============================================================\n");
    printf("  PTO ARM64 Runtime\n");
    printf("============================================================\n");
    
    // Initialize runtime (heap allocated - PTORuntime is too large for stack)
    PTORuntime* rt = (PTORuntime*)calloc(1, sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "Failed to allocate PTORuntime\n");
        return 1;
    }
    pto_runtime_init(rt);
    
    // Allocate test data
    float* input = (float*)calloc(1024 * 1024, sizeof(float));
    float* output = (float*)calloc(1024 * 1024, sizeof(float));
    float* attn_norm_weights = (float*)calloc(1024 * 1024, sizeof(float));
    float* wq = (float*)calloc(1024 * 1024, sizeof(float));
    float* wk = (float*)calloc(1024 * 1024, sizeof(float));
    float* wv = (float*)calloc(1024 * 1024, sizeof(float));
    float* wo = (float*)calloc(1024 * 1024, sizeof(float));
    float* cos_cache = (float*)calloc(1024 * 1024, sizeof(float));
    float* sin_cache = (float*)calloc(1024 * 1024, sizeof(float));
    float* mlp_norm_weights = (float*)calloc(1024 * 1024, sizeof(float));
    float* w_gate = (float*)calloc(1024 * 1024, sizeof(float));
    float* w_up = (float*)calloc(1024 * 1024, sizeof(float));
    float* w_down = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_q_tiles = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_k_tiles = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_v_tiles = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_q_rope = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_k_rope = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_attn_out = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_m_vec = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_l_vec = (float*)calloc(1024 * 1024, sizeof(float));
    float* all_hidden = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_norm = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_scores = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_attn_weights = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_scale = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_gate = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_up = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_swiglu = (float*)calloc(1024 * 1024, sizeof(float));
    float* temp_mlp_out = (float*)calloc(1024 * 1024, sizeof(float));
    float* const_zeros_large = (float*)calloc(1024 * 1024, sizeof(float));
    float* const_zeros_small = (float*)calloc(1024 * 1024, sizeof(float));
    float* const_neg_inf = (float*)calloc(1024 * 1024, sizeof(float));
    int32_t seq_len = 16;  // Default, override with argv[1+arg_offset]
    int32_t tile_rows = 16;  // Default, override with argv[2+arg_offset]
    int32_t num_tiles = 16;  // Default, override with argv[3+arg_offset]
    int32_t zero = 16;  // Default, override with argv[4+arg_offset]

    // Parse command line arguments for integer parameters
    if (argc > 1 + arg_offset) seq_len = atoi(argv[1 + arg_offset]);
    if (argc > 2 + arg_offset) tile_rows = atoi(argv[2 + arg_offset]);
    if (argc > 3 + arg_offset) num_tiles = atoi(argv[3 + arg_offset]);
    if (argc > 4 + arg_offset) zero = atoi(argv[4 + arg_offset]);

    
    if (benchmark_only) {
        // Benchmark mode: only measure orchestration time
        struct timespec start, end;
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        llama_layer_dynamic(rt, input, output, attn_norm_weights, wq, wk, wv, wo, cos_cache, sin_cache, mlp_norm_weights, w_gate, w_up, w_down, all_q_tiles, all_k_tiles, all_v_tiles, all_q_rope, all_k_rope, all_attn_out, all_m_vec, all_l_vec, all_hidden, temp_norm, temp_scores, temp_attn_weights, temp_scale, temp_gate, temp_up, temp_swiglu, temp_mlp_out, const_zeros_large, const_zeros_small, const_neg_inf, seq_len, tile_rows, num_tiles, zero);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
        long long tasks_submitted = rt->total_tasks_scheduled;
        double tasks_per_ms = tasks_submitted / time_ms;
        
        // Output in machine-parseable format
        printf("BENCHMARK: tasks=%lld time_ms=%.3f tasks_per_ms=%.2f\n",
               tasks_submitted, time_ms, tasks_per_ms);
    } else {
        // Normal execution mode
        printf("Running orchestration function: llama_layer_dynamic\n");
        printf("------------------------------------------------------------\n");
        
        llama_layer_dynamic(rt, input, output, attn_norm_weights, wq, wk, wv, wo, cos_cache, sin_cache, mlp_norm_weights, w_gate, w_up, w_down, all_q_tiles, all_k_tiles, all_v_tiles, all_q_rope, all_k_rope, all_attn_out, all_m_vec, all_l_vec, all_hidden, temp_norm, temp_scores, temp_attn_weights, temp_scale, temp_gate, temp_up, temp_swiglu, temp_mlp_out, const_zeros_large, const_zeros_small, const_neg_inf, seq_len, tile_rows, num_tiles, zero);
        
        printf("------------------------------------------------------------\n");
        printf("Submitted %lld tasks\n", (long long)rt->total_tasks_scheduled);
        
        // Execute all tasks
        pto_execute_all(rt);
        
        printf("Execution complete!\n");
    }
    
    // Cleanup - must call shutdown before free to destroy mutexes/condvars
    fflush(stdout);
    pto_runtime_shutdown(rt);
    free(input);
    free(output);
    free(attn_norm_weights);
    free(wq);
    free(wk);
    free(wv);
    free(wo);
    free(cos_cache);
    free(sin_cache);
    free(mlp_norm_weights);
    free(w_gate);
    free(w_up);
    free(w_down);
    free(all_q_tiles);
    free(all_k_tiles);
    free(all_v_tiles);
    free(all_q_rope);
    free(all_k_rope);
    free(all_attn_out);
    free(all_m_vec);
    free(all_l_vec);
    free(all_hidden);
    free(temp_norm);
    free(temp_scores);
    free(temp_attn_weights);
    free(temp_scale);
    free(temp_gate);
    free(temp_up);
    free(temp_swiglu);
    free(temp_mlp_out);
    free(const_zeros_large);
    free(const_zeros_small);
    free(const_neg_inf);
    free(rt);
    
    return 0;
}
