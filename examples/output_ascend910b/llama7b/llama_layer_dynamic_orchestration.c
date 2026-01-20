/**
 * LLaMA 7B Layer Orchestration - seq_len=1024
 * num_tiles: 8
 *
 * THREE-PHASE Task Graph with Cross-Tile Dependencies:
 *
 * Phase 1: Pre-Attention (ALL tiles run in PARALLEL)
 *   - RMSNorm, Q/K/V MatMuls, RoPE for each tile
 *   - No cross-tile dependencies -> massive parallelism
 *
 * Phase 2: Flash Attention (CROSS-TILE dependencies)
 *   - Q[i] attends to ALL K[j], V[j] tiles
 *   - Creates 8x8 attention blocks
 *   - Fan-in pattern: attention output depends on all K,V
 *
 * Phase 3: Post-Attention (after attention completes)
 *   - Output projection, Residual, MLP for each tile
 *   - Parallel across tiles (no cross-tile deps)
 */

#include "pto_runtime.h"
#include "pto_runtime.c"

void build_task_graph(PTORuntime* rt) {
    // Declare per-tile buffers
    // Total tiles: 8
    float input[512];
    float output[512];
    float attn_norm_weights[512];
    float wq[512];
    float wk[512];
    float wv[512];
    float wo[512];
    float cos_cache[512];
    float sin_cache[512];
    float mlp_norm_weights[512];
    float w_gate[512];
    float w_up[512];
    float w_down[512];
    float all_q_tiles[512];
    float all_k_tiles[512];
    float all_v_tiles[512];
    float all_q_rope[512];
    float all_k_rope[512];
    float all_attn_out[512];
    float all_m_vec[512];
    float all_l_vec[512];
    float all_hidden[512];
    float temp_norm[512];
    float temp_scores[512];
    float temp_attn_weights[512];
    float temp_scale[512];
    float temp_gate[512];
    float temp_up[512];
    float temp_swiglu[512];
    float temp_mlp_out[512];
    float const_zeros_large[512];
    float const_zeros_small[512];
    float const_neg_inf[512];

    // ================================================================
    // PHASE 1: Pre-Attention (ALL tiles run in PARALLEL)
    // ================================================================
    // --- Tile 0 Pre-Attention ---
    int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t0, input, 0, 0, 32, 128);
    pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t0, temp_norm, 0, 0, 32, 128);
    pto_task_submit(rt, t0);

    int32_t t1 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t1, temp_norm, 0, 0, 32, 128);
    pto_task_add_input(rt, t1, wq, 0, 0, 32, 128);
    pto_task_add_output(rt, t1, all_q_tiles, 0, 0, 32, 128);
    pto_task_submit(rt, t1);

    int32_t t2 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t2, temp_norm, 0, 0, 32, 128);
    pto_task_add_input(rt, t2, wk, 0, 0, 32, 128);
    pto_task_add_output(rt, t2, all_k_tiles, 0, 0, 32, 128);
    pto_task_submit(rt, t2);

    int32_t t3 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t3, temp_norm, 0, 0, 32, 128);
    pto_task_add_input(rt, t3, wv, 0, 0, 32, 128);
    pto_task_add_output(rt, t3, all_v_tiles, 0, 0, 32, 128);
    pto_task_submit(rt, t3);

    int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t4, all_q_tiles, 0, 0, 32, 128);
    pto_task_add_output(rt, t4, all_q_rope, 0, 0, 32, 128);
    pto_task_submit(rt, t4);

    int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t5, all_k_tiles, 0, 0, 32, 128);
    pto_task_add_output(rt, t5, all_k_rope, 0, 0, 32, 128);
    pto_task_submit(rt, t5);

    // --- Tile 1 Pre-Attention ---
    int32_t t6 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t6, input, 32, 0, 32, 128);
    pto_task_add_input(rt, t6, attn_norm_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t6, temp_norm, 32, 0, 32, 128);
    pto_task_submit(rt, t6);

    int32_t t7 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t7, temp_norm, 32, 0, 32, 128);
    pto_task_add_input(rt, t7, wq, 32, 0, 32, 128);
    pto_task_add_output(rt, t7, all_q_tiles, 32, 0, 32, 128);
    pto_task_submit(rt, t7);

    int32_t t8 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t8, temp_norm, 32, 0, 32, 128);
    pto_task_add_input(rt, t8, wk, 32, 0, 32, 128);
    pto_task_add_output(rt, t8, all_k_tiles, 32, 0, 32, 128);
    pto_task_submit(rt, t8);

    int32_t t9 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t9, temp_norm, 32, 0, 32, 128);
    pto_task_add_input(rt, t9, wv, 32, 0, 32, 128);
    pto_task_add_output(rt, t9, all_v_tiles, 32, 0, 32, 128);
    pto_task_submit(rt, t9);

    int32_t t10 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t10, all_q_tiles, 32, 0, 32, 128);
    pto_task_add_output(rt, t10, all_q_rope, 32, 0, 32, 128);
    pto_task_submit(rt, t10);

    int32_t t11 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t11, all_k_tiles, 32, 0, 32, 128);
    pto_task_add_output(rt, t11, all_k_rope, 32, 0, 32, 128);
    pto_task_submit(rt, t11);

    // --- Tile 2 Pre-Attention ---
    int32_t t12 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t12, input, 64, 0, 32, 128);
    pto_task_add_input(rt, t12, attn_norm_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t12, temp_norm, 64, 0, 32, 128);
    pto_task_submit(rt, t12);

    int32_t t13 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t13, temp_norm, 64, 0, 32, 128);
    pto_task_add_input(rt, t13, wq, 64, 0, 32, 128);
    pto_task_add_output(rt, t13, all_q_tiles, 64, 0, 32, 128);
    pto_task_submit(rt, t13);

    int32_t t14 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t14, temp_norm, 64, 0, 32, 128);
    pto_task_add_input(rt, t14, wk, 64, 0, 32, 128);
    pto_task_add_output(rt, t14, all_k_tiles, 64, 0, 32, 128);
    pto_task_submit(rt, t14);

    int32_t t15 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t15, temp_norm, 64, 0, 32, 128);
    pto_task_add_input(rt, t15, wv, 64, 0, 32, 128);
    pto_task_add_output(rt, t15, all_v_tiles, 64, 0, 32, 128);
    pto_task_submit(rt, t15);

    int32_t t16 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t16, all_q_tiles, 64, 0, 32, 128);
    pto_task_add_output(rt, t16, all_q_rope, 64, 0, 32, 128);
    pto_task_submit(rt, t16);

    int32_t t17 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t17, all_k_tiles, 64, 0, 32, 128);
    pto_task_add_output(rt, t17, all_k_rope, 64, 0, 32, 128);
    pto_task_submit(rt, t17);

    // --- Tile 3 Pre-Attention ---
    int32_t t18 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t18, input, 96, 0, 32, 128);
    pto_task_add_input(rt, t18, attn_norm_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t18, temp_norm, 96, 0, 32, 128);
    pto_task_submit(rt, t18);

    int32_t t19 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t19, temp_norm, 96, 0, 32, 128);
    pto_task_add_input(rt, t19, wq, 96, 0, 32, 128);
    pto_task_add_output(rt, t19, all_q_tiles, 96, 0, 32, 128);
    pto_task_submit(rt, t19);

    int32_t t20 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t20, temp_norm, 96, 0, 32, 128);
    pto_task_add_input(rt, t20, wk, 96, 0, 32, 128);
    pto_task_add_output(rt, t20, all_k_tiles, 96, 0, 32, 128);
    pto_task_submit(rt, t20);

    int32_t t21 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t21, temp_norm, 96, 0, 32, 128);
    pto_task_add_input(rt, t21, wv, 96, 0, 32, 128);
    pto_task_add_output(rt, t21, all_v_tiles, 96, 0, 32, 128);
    pto_task_submit(rt, t21);

    int32_t t22 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t22, all_q_tiles, 96, 0, 32, 128);
    pto_task_add_output(rt, t22, all_q_rope, 96, 0, 32, 128);
    pto_task_submit(rt, t22);

    int32_t t23 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t23, all_k_tiles, 96, 0, 32, 128);
    pto_task_add_output(rt, t23, all_k_rope, 96, 0, 32, 128);
    pto_task_submit(rt, t23);

    // --- Tile 4 Pre-Attention ---
    int32_t t24 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t24, input, 128, 0, 32, 128);
    pto_task_add_input(rt, t24, attn_norm_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t24, temp_norm, 128, 0, 32, 128);
    pto_task_submit(rt, t24);

    int32_t t25 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t25, temp_norm, 128, 0, 32, 128);
    pto_task_add_input(rt, t25, wq, 128, 0, 32, 128);
    pto_task_add_output(rt, t25, all_q_tiles, 128, 0, 32, 128);
    pto_task_submit(rt, t25);

    int32_t t26 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t26, temp_norm, 128, 0, 32, 128);
    pto_task_add_input(rt, t26, wk, 128, 0, 32, 128);
    pto_task_add_output(rt, t26, all_k_tiles, 128, 0, 32, 128);
    pto_task_submit(rt, t26);

    int32_t t27 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t27, temp_norm, 128, 0, 32, 128);
    pto_task_add_input(rt, t27, wv, 128, 0, 32, 128);
    pto_task_add_output(rt, t27, all_v_tiles, 128, 0, 32, 128);
    pto_task_submit(rt, t27);

    int32_t t28 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t28, all_q_tiles, 128, 0, 32, 128);
    pto_task_add_output(rt, t28, all_q_rope, 128, 0, 32, 128);
    pto_task_submit(rt, t28);

    int32_t t29 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t29, all_k_tiles, 128, 0, 32, 128);
    pto_task_add_output(rt, t29, all_k_rope, 128, 0, 32, 128);
    pto_task_submit(rt, t29);

    // --- Tile 5 Pre-Attention ---
    int32_t t30 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t30, input, 160, 0, 32, 128);
    pto_task_add_input(rt, t30, attn_norm_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t30, temp_norm, 160, 0, 32, 128);
    pto_task_submit(rt, t30);

    int32_t t31 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t31, temp_norm, 160, 0, 32, 128);
    pto_task_add_input(rt, t31, wq, 160, 0, 32, 128);
    pto_task_add_output(rt, t31, all_q_tiles, 160, 0, 32, 128);
    pto_task_submit(rt, t31);

    int32_t t32 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t32, temp_norm, 160, 0, 32, 128);
    pto_task_add_input(rt, t32, wk, 160, 0, 32, 128);
    pto_task_add_output(rt, t32, all_k_tiles, 160, 0, 32, 128);
    pto_task_submit(rt, t32);

    int32_t t33 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t33, temp_norm, 160, 0, 32, 128);
    pto_task_add_input(rt, t33, wv, 160, 0, 32, 128);
    pto_task_add_output(rt, t33, all_v_tiles, 160, 0, 32, 128);
    pto_task_submit(rt, t33);

    int32_t t34 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t34, all_q_tiles, 160, 0, 32, 128);
    pto_task_add_output(rt, t34, all_q_rope, 160, 0, 32, 128);
    pto_task_submit(rt, t34);

    int32_t t35 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t35, all_k_tiles, 160, 0, 32, 128);
    pto_task_add_output(rt, t35, all_k_rope, 160, 0, 32, 128);
    pto_task_submit(rt, t35);

    // --- Tile 6 Pre-Attention ---
    int32_t t36 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t36, input, 192, 0, 32, 128);
    pto_task_add_input(rt, t36, attn_norm_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t36, temp_norm, 192, 0, 32, 128);
    pto_task_submit(rt, t36);

    int32_t t37 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t37, temp_norm, 192, 0, 32, 128);
    pto_task_add_input(rt, t37, wq, 192, 0, 32, 128);
    pto_task_add_output(rt, t37, all_q_tiles, 192, 0, 32, 128);
    pto_task_submit(rt, t37);

    int32_t t38 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t38, temp_norm, 192, 0, 32, 128);
    pto_task_add_input(rt, t38, wk, 192, 0, 32, 128);
    pto_task_add_output(rt, t38, all_k_tiles, 192, 0, 32, 128);
    pto_task_submit(rt, t38);

    int32_t t39 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t39, temp_norm, 192, 0, 32, 128);
    pto_task_add_input(rt, t39, wv, 192, 0, 32, 128);
    pto_task_add_output(rt, t39, all_v_tiles, 192, 0, 32, 128);
    pto_task_submit(rt, t39);

    int32_t t40 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t40, all_q_tiles, 192, 0, 32, 128);
    pto_task_add_output(rt, t40, all_q_rope, 192, 0, 32, 128);
    pto_task_submit(rt, t40);

    int32_t t41 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t41, all_k_tiles, 192, 0, 32, 128);
    pto_task_add_output(rt, t41, all_k_rope, 192, 0, 32, 128);
    pto_task_submit(rt, t41);

    // --- Tile 7 Pre-Attention ---
    int32_t t42 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t42, input, 224, 0, 32, 128);
    pto_task_add_input(rt, t42, attn_norm_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t42, temp_norm, 224, 0, 32, 128);
    pto_task_submit(rt, t42);

    int32_t t43 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t43, temp_norm, 224, 0, 32, 128);
    pto_task_add_input(rt, t43, wq, 224, 0, 32, 128);
    pto_task_add_output(rt, t43, all_q_tiles, 224, 0, 32, 128);
    pto_task_submit(rt, t43);

    int32_t t44 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t44, temp_norm, 224, 0, 32, 128);
    pto_task_add_input(rt, t44, wk, 224, 0, 32, 128);
    pto_task_add_output(rt, t44, all_k_tiles, 224, 0, 32, 128);
    pto_task_submit(rt, t44);

    int32_t t45 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t45, temp_norm, 224, 0, 32, 128);
    pto_task_add_input(rt, t45, wv, 224, 0, 32, 128);
    pto_task_add_output(rt, t45, all_v_tiles, 224, 0, 32, 128);
    pto_task_submit(rt, t45);

    int32_t t46 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t46, all_q_tiles, 224, 0, 32, 128);
    pto_task_add_output(rt, t46, all_q_rope, 224, 0, 32, 128);
    pto_task_submit(rt, t46);

    int32_t t47 = pto_task_alloc(rt, "rope_tile", NULL, 98304, 65536);
    pto_task_add_input(rt, t47, all_k_tiles, 224, 0, 32, 128);
    pto_task_add_output(rt, t47, all_k_rope, 224, 0, 32, 128);
    pto_task_submit(rt, t47);


    // ================================================================
    // PHASE 2: Flash Attention (CROSS-TILE dependencies)
    // Each Q tile attends to ALL 8 K,V tiles
    // ================================================================
    // --- Q tile 0 attending to all K,V tiles ---
    int32_t t48 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t48, all_attn_out, 0, 0, 32, 128);
    pto_task_add_output(rt, t48, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t48, all_m_vec, 0, 0, 32, 128);
    pto_task_submit(rt, t48);

    // Cross-tile: Q[0] x K/V[0]
    int32_t t49 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t49, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t49, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t49, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t49);

    int32_t t50 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t50, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t50, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t50, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t50, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t50, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t50, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t50, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t50);

    // Cross-tile: Q[0] x K/V[0]
    int32_t t51 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t51, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t51, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t51, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t51, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t51, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t51);

    // Cross-tile: Q[0] x K/V[1]
    int32_t t52 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t52, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t52, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t52, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t52);

    int32_t t53 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t53, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t53, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t53, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t53, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t53, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t53, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t53, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t53);

    // Cross-tile: Q[0] x K/V[1]
    int32_t t54 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t54, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t54, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t54, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t54, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t54, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t54);

    // Cross-tile: Q[0] x K/V[2]
    int32_t t55 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t55, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t55, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t55, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t55);

    int32_t t56 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t56, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t56, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t56, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t56, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t56, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t56, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t56, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t56);

    // Cross-tile: Q[0] x K/V[2]
    int32_t t57 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t57, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t57, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t57, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t57, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t57, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t57);

    // Cross-tile: Q[0] x K/V[3]
    int32_t t58 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t58, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t58, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t58, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t58);

    int32_t t59 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t59, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t59, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t59, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t59, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t59, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t59, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t59, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t59);

    // Cross-tile: Q[0] x K/V[3]
    int32_t t60 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t60, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t60, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t60, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t60, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t60, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t60);

    // Cross-tile: Q[0] x K/V[4]
    int32_t t61 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t61, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t61, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t61, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t61);

    int32_t t62 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t62, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t62, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t62, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t62, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t62, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t62, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t62, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t62);

    // Cross-tile: Q[0] x K/V[4]
    int32_t t63 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t63, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t63, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t63, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t63, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t63, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t63);

    // Cross-tile: Q[0] x K/V[5]
    int32_t t64 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t64, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t64, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t64, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t64);

    int32_t t65 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t65, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t65, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t65, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t65, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t65, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t65, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t65, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t65);

    // Cross-tile: Q[0] x K/V[5]
    int32_t t66 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t66, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t66, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t66, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t66, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t66, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t66);

    // Cross-tile: Q[0] x K/V[6]
    int32_t t67 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t67, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t67, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t67, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t67);

    int32_t t68 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t68, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t68, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t68, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t68, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t68, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t68, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t68, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t68);

    // Cross-tile: Q[0] x K/V[6]
    int32_t t69 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t69, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t69, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t69, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t69, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t69, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t69);

    // Cross-tile: Q[0] x K/V[7]
    int32_t t70 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t70, all_q_rope, 0, 0, 32, 128);
    pto_task_add_input(rt, t70, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t70, temp_scores, 0, 0, 32, 128);
    pto_task_submit(rt, t70);

    int32_t t71 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t71, temp_scores, 0, 0, 32, 128);
    pto_task_add_input(rt, t71, all_m_vec, 0, 0, 32, 128);
    pto_task_add_input(rt, t71, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t71, all_m_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t71, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t71, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t71, temp_scale, 0, 0, 32, 128);
    pto_task_submit(rt, t71);

    // Cross-tile: Q[0] x K/V[7]
    int32_t t72 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t72, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t72, temp_attn_weights, 0, 0, 32, 128);
    pto_task_add_input(rt, t72, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t72, temp_scale, 0, 0, 32, 128);
    pto_task_add_output(rt, t72, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t72);

    int32_t t73 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t73, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t73, all_l_vec, 0, 0, 32, 128);
    pto_task_add_output(rt, t73, all_attn_out, 0, 0, 32, 128);
    pto_task_submit(rt, t73);

    // --- Q tile 1 attending to all K,V tiles ---
    int32_t t74 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t74, all_attn_out, 32, 0, 32, 128);
    pto_task_add_output(rt, t74, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t74, all_m_vec, 32, 0, 32, 128);
    pto_task_submit(rt, t74);

    // Cross-tile: Q[1] x K/V[0]
    int32_t t75 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t75, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t75, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t75, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t75);

    int32_t t76 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t76, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t76, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t76, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t76, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t76, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t76, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t76, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t76);

    // Cross-tile: Q[1] x K/V[0]
    int32_t t77 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t77, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t77, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t77, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t77, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t77, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t77);

    // Cross-tile: Q[1] x K/V[1]
    int32_t t78 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t78, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t78, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t78, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t78);

    int32_t t79 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t79, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t79, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t79, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t79, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t79, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t79, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t79, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t79);

    // Cross-tile: Q[1] x K/V[1]
    int32_t t80 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t80, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t80, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t80, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t80, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t80, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t80);

    // Cross-tile: Q[1] x K/V[2]
    int32_t t81 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t81, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t81, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t81, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t81);

    int32_t t82 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t82, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t82, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t82, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t82, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t82, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t82, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t82, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t82);

    // Cross-tile: Q[1] x K/V[2]
    int32_t t83 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t83, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t83, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t83, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t83, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t83, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t83);

    // Cross-tile: Q[1] x K/V[3]
    int32_t t84 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t84, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t84, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t84, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t84);

    int32_t t85 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t85, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t85, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t85, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t85, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t85, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t85, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t85, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t85);

    // Cross-tile: Q[1] x K/V[3]
    int32_t t86 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t86, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t86, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t86, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t86, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t86, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t86);

    // Cross-tile: Q[1] x K/V[4]
    int32_t t87 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t87, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t87, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t87, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t87);

    int32_t t88 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t88, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t88, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t88, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t88, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t88, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t88, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t88, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t88);

    // Cross-tile: Q[1] x K/V[4]
    int32_t t89 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t89, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t89, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t89, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t89, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t89, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t89);

    // Cross-tile: Q[1] x K/V[5]
    int32_t t90 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t90, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t90, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t90, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t90);

    int32_t t91 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t91, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t91, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t91, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t91, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t91, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t91, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t91, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t91);

    // Cross-tile: Q[1] x K/V[5]
    int32_t t92 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t92, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t92, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t92, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t92, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t92, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t92);

    // Cross-tile: Q[1] x K/V[6]
    int32_t t93 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t93, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t93, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t93, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t93);

    int32_t t94 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t94, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t94, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t94, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t94, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t94, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t94, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t94, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t94);

    // Cross-tile: Q[1] x K/V[6]
    int32_t t95 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t95, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t95, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t95, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t95, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t95, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t95);

    // Cross-tile: Q[1] x K/V[7]
    int32_t t96 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t96, all_q_rope, 32, 0, 32, 128);
    pto_task_add_input(rt, t96, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t96, temp_scores, 32, 0, 32, 128);
    pto_task_submit(rt, t96);

    int32_t t97 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t97, temp_scores, 32, 0, 32, 128);
    pto_task_add_input(rt, t97, all_m_vec, 32, 0, 32, 128);
    pto_task_add_input(rt, t97, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t97, all_m_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t97, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t97, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t97, temp_scale, 32, 0, 32, 128);
    pto_task_submit(rt, t97);

    // Cross-tile: Q[1] x K/V[7]
    int32_t t98 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t98, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t98, temp_attn_weights, 32, 0, 32, 128);
    pto_task_add_input(rt, t98, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t98, temp_scale, 32, 0, 32, 128);
    pto_task_add_output(rt, t98, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t98);

    int32_t t99 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t99, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t99, all_l_vec, 32, 0, 32, 128);
    pto_task_add_output(rt, t99, all_attn_out, 32, 0, 32, 128);
    pto_task_submit(rt, t99);

    // --- Q tile 2 attending to all K,V tiles ---
    int32_t t100 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t100, all_attn_out, 64, 0, 32, 128);
    pto_task_add_output(rt, t100, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t100, all_m_vec, 64, 0, 32, 128);
    pto_task_submit(rt, t100);

    // Cross-tile: Q[2] x K/V[0]
    int32_t t101 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t101, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t101, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t101, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t101);

    int32_t t102 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t102, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t102, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t102, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t102, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t102, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t102, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t102, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t102);

    // Cross-tile: Q[2] x K/V[0]
    int32_t t103 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t103, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t103, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t103, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t103, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t103, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t103);

    // Cross-tile: Q[2] x K/V[1]
    int32_t t104 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t104, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t104, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t104, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t104);

    int32_t t105 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t105, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t105, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t105, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t105, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t105, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t105, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t105, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t105);

    // Cross-tile: Q[2] x K/V[1]
    int32_t t106 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t106, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t106, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t106, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t106, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t106, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t106);

    // Cross-tile: Q[2] x K/V[2]
    int32_t t107 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t107, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t107, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t107, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t107);

    int32_t t108 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t108, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t108, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t108, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t108, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t108, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t108, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t108, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t108);

    // Cross-tile: Q[2] x K/V[2]
    int32_t t109 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t109, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t109, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t109, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t109, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t109, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t109);

    // Cross-tile: Q[2] x K/V[3]
    int32_t t110 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t110, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t110, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t110, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t110);

    int32_t t111 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t111, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t111, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t111, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t111, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t111, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t111, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t111, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t111);

    // Cross-tile: Q[2] x K/V[3]
    int32_t t112 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t112, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t112, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t112, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t112, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t112, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t112);

    // Cross-tile: Q[2] x K/V[4]
    int32_t t113 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t113, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t113, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t113, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t113);

    int32_t t114 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t114, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t114, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t114, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t114, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t114, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t114, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t114, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t114);

    // Cross-tile: Q[2] x K/V[4]
    int32_t t115 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t115, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t115, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t115, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t115, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t115, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t115);

    // Cross-tile: Q[2] x K/V[5]
    int32_t t116 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t116, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t116, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t116, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t116);

    int32_t t117 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t117, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t117, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t117, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t117, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t117, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t117, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t117, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t117);

    // Cross-tile: Q[2] x K/V[5]
    int32_t t118 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t118, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t118, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t118, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t118, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t118, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t118);

    // Cross-tile: Q[2] x K/V[6]
    int32_t t119 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t119, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t119, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t119, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t119);

    int32_t t120 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t120, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t120, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t120, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t120, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t120, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t120, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t120, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t120);

    // Cross-tile: Q[2] x K/V[6]
    int32_t t121 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t121, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t121, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t121, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t121, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t121, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t121);

    // Cross-tile: Q[2] x K/V[7]
    int32_t t122 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t122, all_q_rope, 64, 0, 32, 128);
    pto_task_add_input(rt, t122, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t122, temp_scores, 64, 0, 32, 128);
    pto_task_submit(rt, t122);

    int32_t t123 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t123, temp_scores, 64, 0, 32, 128);
    pto_task_add_input(rt, t123, all_m_vec, 64, 0, 32, 128);
    pto_task_add_input(rt, t123, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t123, all_m_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t123, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t123, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t123, temp_scale, 64, 0, 32, 128);
    pto_task_submit(rt, t123);

    // Cross-tile: Q[2] x K/V[7]
    int32_t t124 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t124, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t124, temp_attn_weights, 64, 0, 32, 128);
    pto_task_add_input(rt, t124, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t124, temp_scale, 64, 0, 32, 128);
    pto_task_add_output(rt, t124, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t124);

    int32_t t125 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t125, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t125, all_l_vec, 64, 0, 32, 128);
    pto_task_add_output(rt, t125, all_attn_out, 64, 0, 32, 128);
    pto_task_submit(rt, t125);

    // --- Q tile 3 attending to all K,V tiles ---
    int32_t t126 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t126, all_attn_out, 96, 0, 32, 128);
    pto_task_add_output(rt, t126, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t126, all_m_vec, 96, 0, 32, 128);
    pto_task_submit(rt, t126);

    // Cross-tile: Q[3] x K/V[0]
    int32_t t127 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t127, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t127, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t127, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t127);

    int32_t t128 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t128, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t128, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t128, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t128, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t128, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t128, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t128, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t128);

    // Cross-tile: Q[3] x K/V[0]
    int32_t t129 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t129, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t129, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t129, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t129, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t129, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t129);

    // Cross-tile: Q[3] x K/V[1]
    int32_t t130 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t130, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t130, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t130, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t130);

    int32_t t131 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t131, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t131, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t131, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t131, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t131, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t131, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t131, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t131);

    // Cross-tile: Q[3] x K/V[1]
    int32_t t132 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t132, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t132, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t132, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t132, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t132, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t132);

    // Cross-tile: Q[3] x K/V[2]
    int32_t t133 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t133, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t133, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t133, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t133);

    int32_t t134 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t134, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t134, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t134, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t134, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t134, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t134, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t134, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t134);

    // Cross-tile: Q[3] x K/V[2]
    int32_t t135 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t135, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t135, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t135, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t135, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t135, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t135);

    // Cross-tile: Q[3] x K/V[3]
    int32_t t136 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t136, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t136, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t136, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t136);

    int32_t t137 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t137, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t137, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t137, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t137, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t137, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t137, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t137, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t137);

    // Cross-tile: Q[3] x K/V[3]
    int32_t t138 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t138, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t138, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t138, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t138, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t138, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t138);

    // Cross-tile: Q[3] x K/V[4]
    int32_t t139 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t139, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t139, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t139, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t139);

    int32_t t140 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t140, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t140, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t140, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t140, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t140, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t140, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t140, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t140);

    // Cross-tile: Q[3] x K/V[4]
    int32_t t141 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t141, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t141, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t141, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t141, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t141, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t141);

    // Cross-tile: Q[3] x K/V[5]
    int32_t t142 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t142, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t142, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t142, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t142);

    int32_t t143 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t143, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t143, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t143, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t143, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t143, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t143, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t143, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t143);

    // Cross-tile: Q[3] x K/V[5]
    int32_t t144 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t144, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t144, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t144, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t144, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t144, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t144);

    // Cross-tile: Q[3] x K/V[6]
    int32_t t145 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t145, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t145, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t145, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t145);

    int32_t t146 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t146, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t146, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t146, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t146, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t146, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t146, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t146, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t146);

    // Cross-tile: Q[3] x K/V[6]
    int32_t t147 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t147, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t147, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t147, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t147, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t147, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t147);

    // Cross-tile: Q[3] x K/V[7]
    int32_t t148 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t148, all_q_rope, 96, 0, 32, 128);
    pto_task_add_input(rt, t148, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t148, temp_scores, 96, 0, 32, 128);
    pto_task_submit(rt, t148);

    int32_t t149 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t149, temp_scores, 96, 0, 32, 128);
    pto_task_add_input(rt, t149, all_m_vec, 96, 0, 32, 128);
    pto_task_add_input(rt, t149, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t149, all_m_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t149, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t149, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t149, temp_scale, 96, 0, 32, 128);
    pto_task_submit(rt, t149);

    // Cross-tile: Q[3] x K/V[7]
    int32_t t150 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t150, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t150, temp_attn_weights, 96, 0, 32, 128);
    pto_task_add_input(rt, t150, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t150, temp_scale, 96, 0, 32, 128);
    pto_task_add_output(rt, t150, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t150);

    int32_t t151 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t151, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t151, all_l_vec, 96, 0, 32, 128);
    pto_task_add_output(rt, t151, all_attn_out, 96, 0, 32, 128);
    pto_task_submit(rt, t151);

    // --- Q tile 4 attending to all K,V tiles ---
    int32_t t152 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t152, all_attn_out, 128, 0, 32, 128);
    pto_task_add_output(rt, t152, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t152, all_m_vec, 128, 0, 32, 128);
    pto_task_submit(rt, t152);

    // Cross-tile: Q[4] x K/V[0]
    int32_t t153 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t153, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t153, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t153, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t153);

    int32_t t154 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t154, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t154, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t154, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t154, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t154, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t154, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t154, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t154);

    // Cross-tile: Q[4] x K/V[0]
    int32_t t155 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t155, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t155, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t155, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t155, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t155, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t155);

    // Cross-tile: Q[4] x K/V[1]
    int32_t t156 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t156, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t156, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t156, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t156);

    int32_t t157 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t157, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t157, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t157, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t157, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t157, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t157, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t157, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t157);

    // Cross-tile: Q[4] x K/V[1]
    int32_t t158 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t158, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t158, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t158, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t158, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t158, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t158);

    // Cross-tile: Q[4] x K/V[2]
    int32_t t159 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t159, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t159, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t159, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t159);

    int32_t t160 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t160, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t160, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t160, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t160, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t160, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t160, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t160, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t160);

    // Cross-tile: Q[4] x K/V[2]
    int32_t t161 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t161, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t161, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t161, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t161, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t161, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t161);

    // Cross-tile: Q[4] x K/V[3]
    int32_t t162 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t162, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t162, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t162, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t162);

    int32_t t163 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t163, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t163, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t163, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t163, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t163, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t163, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t163, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t163);

    // Cross-tile: Q[4] x K/V[3]
    int32_t t164 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t164, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t164, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t164, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t164, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t164, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t164);

    // Cross-tile: Q[4] x K/V[4]
    int32_t t165 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t165, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t165, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t165, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t165);

    int32_t t166 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t166, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t166, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t166, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t166, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t166, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t166, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t166, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t166);

    // Cross-tile: Q[4] x K/V[4]
    int32_t t167 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t167, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t167, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t167, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t167, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t167, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t167);

    // Cross-tile: Q[4] x K/V[5]
    int32_t t168 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t168, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t168, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t168, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t168);

    int32_t t169 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t169, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t169, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t169, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t169, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t169, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t169, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t169, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t169);

    // Cross-tile: Q[4] x K/V[5]
    int32_t t170 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t170, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t170, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t170, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t170, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t170, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t170);

    // Cross-tile: Q[4] x K/V[6]
    int32_t t171 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t171, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t171, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t171, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t171);

    int32_t t172 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t172, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t172, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t172, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t172, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t172, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t172, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t172, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t172);

    // Cross-tile: Q[4] x K/V[6]
    int32_t t173 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t173, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t173, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t173, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t173, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t173, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t173);

    // Cross-tile: Q[4] x K/V[7]
    int32_t t174 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t174, all_q_rope, 128, 0, 32, 128);
    pto_task_add_input(rt, t174, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t174, temp_scores, 128, 0, 32, 128);
    pto_task_submit(rt, t174);

    int32_t t175 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t175, temp_scores, 128, 0, 32, 128);
    pto_task_add_input(rt, t175, all_m_vec, 128, 0, 32, 128);
    pto_task_add_input(rt, t175, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t175, all_m_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t175, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t175, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t175, temp_scale, 128, 0, 32, 128);
    pto_task_submit(rt, t175);

    // Cross-tile: Q[4] x K/V[7]
    int32_t t176 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t176, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t176, temp_attn_weights, 128, 0, 32, 128);
    pto_task_add_input(rt, t176, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t176, temp_scale, 128, 0, 32, 128);
    pto_task_add_output(rt, t176, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t176);

    int32_t t177 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t177, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t177, all_l_vec, 128, 0, 32, 128);
    pto_task_add_output(rt, t177, all_attn_out, 128, 0, 32, 128);
    pto_task_submit(rt, t177);

    // --- Q tile 5 attending to all K,V tiles ---
    int32_t t178 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t178, all_attn_out, 160, 0, 32, 128);
    pto_task_add_output(rt, t178, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t178, all_m_vec, 160, 0, 32, 128);
    pto_task_submit(rt, t178);

    // Cross-tile: Q[5] x K/V[0]
    int32_t t179 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t179, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t179, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t179, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t179);

    int32_t t180 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t180, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t180, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t180, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t180, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t180, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t180, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t180, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t180);

    // Cross-tile: Q[5] x K/V[0]
    int32_t t181 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t181, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t181, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t181, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t181, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t181, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t181);

    // Cross-tile: Q[5] x K/V[1]
    int32_t t182 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t182, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t182, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t182, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t182);

    int32_t t183 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t183, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t183, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t183, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t183, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t183, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t183, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t183, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t183);

    // Cross-tile: Q[5] x K/V[1]
    int32_t t184 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t184, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t184, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t184, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t184, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t184, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t184);

    // Cross-tile: Q[5] x K/V[2]
    int32_t t185 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t185, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t185, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t185, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t185);

    int32_t t186 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t186, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t186, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t186, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t186, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t186, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t186, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t186, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t186);

    // Cross-tile: Q[5] x K/V[2]
    int32_t t187 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t187, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t187, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t187, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t187, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t187, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t187);

    // Cross-tile: Q[5] x K/V[3]
    int32_t t188 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t188, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t188, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t188, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t188);

    int32_t t189 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t189, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t189, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t189, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t189, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t189, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t189, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t189, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t189);

    // Cross-tile: Q[5] x K/V[3]
    int32_t t190 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t190, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t190, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t190, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t190, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t190, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t190);

    // Cross-tile: Q[5] x K/V[4]
    int32_t t191 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t191, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t191, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t191, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t191);

    int32_t t192 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t192, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t192, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t192, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t192, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t192, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t192, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t192, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t192);

    // Cross-tile: Q[5] x K/V[4]
    int32_t t193 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t193, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t193, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t193, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t193, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t193, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t193);

    // Cross-tile: Q[5] x K/V[5]
    int32_t t194 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t194, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t194, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t194, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t194);

    int32_t t195 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t195, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t195, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t195, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t195, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t195, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t195, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t195, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t195);

    // Cross-tile: Q[5] x K/V[5]
    int32_t t196 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t196, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t196, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t196, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t196, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t196, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t196);

    // Cross-tile: Q[5] x K/V[6]
    int32_t t197 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t197, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t197, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t197, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t197);

    int32_t t198 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t198, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t198, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t198, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t198, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t198, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t198, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t198, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t198);

    // Cross-tile: Q[5] x K/V[6]
    int32_t t199 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t199, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t199, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t199, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t199, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t199, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t199);

    // Cross-tile: Q[5] x K/V[7]
    int32_t t200 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t200, all_q_rope, 160, 0, 32, 128);
    pto_task_add_input(rt, t200, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t200, temp_scores, 160, 0, 32, 128);
    pto_task_submit(rt, t200);

    int32_t t201 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t201, temp_scores, 160, 0, 32, 128);
    pto_task_add_input(rt, t201, all_m_vec, 160, 0, 32, 128);
    pto_task_add_input(rt, t201, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t201, all_m_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t201, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t201, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t201, temp_scale, 160, 0, 32, 128);
    pto_task_submit(rt, t201);

    // Cross-tile: Q[5] x K/V[7]
    int32_t t202 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t202, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t202, temp_attn_weights, 160, 0, 32, 128);
    pto_task_add_input(rt, t202, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t202, temp_scale, 160, 0, 32, 128);
    pto_task_add_output(rt, t202, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t202);

    int32_t t203 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t203, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t203, all_l_vec, 160, 0, 32, 128);
    pto_task_add_output(rt, t203, all_attn_out, 160, 0, 32, 128);
    pto_task_submit(rt, t203);

    // --- Q tile 6 attending to all K,V tiles ---
    int32_t t204 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t204, all_attn_out, 192, 0, 32, 128);
    pto_task_add_output(rt, t204, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t204, all_m_vec, 192, 0, 32, 128);
    pto_task_submit(rt, t204);

    // Cross-tile: Q[6] x K/V[0]
    int32_t t205 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t205, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t205, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t205, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t205);

    int32_t t206 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t206, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t206, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t206, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t206, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t206, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t206, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t206, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t206);

    // Cross-tile: Q[6] x K/V[0]
    int32_t t207 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t207, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t207, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t207, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t207, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t207, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t207);

    // Cross-tile: Q[6] x K/V[1]
    int32_t t208 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t208, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t208, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t208, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t208);

    int32_t t209 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t209, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t209, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t209, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t209, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t209, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t209, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t209, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t209);

    // Cross-tile: Q[6] x K/V[1]
    int32_t t210 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t210, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t210, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t210, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t210, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t210, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t210);

    // Cross-tile: Q[6] x K/V[2]
    int32_t t211 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t211, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t211, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t211, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t211);

    int32_t t212 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t212, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t212, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t212, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t212, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t212, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t212, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t212, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t212);

    // Cross-tile: Q[6] x K/V[2]
    int32_t t213 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t213, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t213, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t213, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t213, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t213, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t213);

    // Cross-tile: Q[6] x K/V[3]
    int32_t t214 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t214, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t214, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t214, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t214);

    int32_t t215 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t215, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t215, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t215, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t215, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t215, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t215, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t215, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t215);

    // Cross-tile: Q[6] x K/V[3]
    int32_t t216 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t216, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t216, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t216, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t216, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t216, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t216);

    // Cross-tile: Q[6] x K/V[4]
    int32_t t217 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t217, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t217, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t217, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t217);

    int32_t t218 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t218, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t218, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t218, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t218, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t218, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t218, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t218, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t218);

    // Cross-tile: Q[6] x K/V[4]
    int32_t t219 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t219, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t219, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t219, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t219, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t219, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t219);

    // Cross-tile: Q[6] x K/V[5]
    int32_t t220 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t220, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t220, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t220, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t220);

    int32_t t221 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t221, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t221, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t221, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t221, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t221, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t221, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t221, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t221);

    // Cross-tile: Q[6] x K/V[5]
    int32_t t222 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t222, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t222, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t222, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t222, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t222, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t222);

    // Cross-tile: Q[6] x K/V[6]
    int32_t t223 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t223, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t223, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t223, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t223);

    int32_t t224 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t224, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t224, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t224, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t224, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t224, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t224, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t224, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t224);

    // Cross-tile: Q[6] x K/V[6]
    int32_t t225 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t225, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t225, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t225, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t225, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t225, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t225);

    // Cross-tile: Q[6] x K/V[7]
    int32_t t226 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t226, all_q_rope, 192, 0, 32, 128);
    pto_task_add_input(rt, t226, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t226, temp_scores, 192, 0, 32, 128);
    pto_task_submit(rt, t226);

    int32_t t227 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t227, temp_scores, 192, 0, 32, 128);
    pto_task_add_input(rt, t227, all_m_vec, 192, 0, 32, 128);
    pto_task_add_input(rt, t227, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t227, all_m_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t227, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t227, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t227, temp_scale, 192, 0, 32, 128);
    pto_task_submit(rt, t227);

    // Cross-tile: Q[6] x K/V[7]
    int32_t t228 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t228, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t228, temp_attn_weights, 192, 0, 32, 128);
    pto_task_add_input(rt, t228, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t228, temp_scale, 192, 0, 32, 128);
    pto_task_add_output(rt, t228, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t228);

    int32_t t229 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t229, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t229, all_l_vec, 192, 0, 32, 128);
    pto_task_add_output(rt, t229, all_attn_out, 192, 0, 32, 128);
    pto_task_submit(rt, t229);

    // --- Q tile 7 attending to all K,V tiles ---
    int32_t t230 = pto_task_alloc(rt, "flash_attn_init_state", NULL, 33280, 33280);
    pto_task_add_output(rt, t230, all_attn_out, 224, 0, 32, 128);
    pto_task_add_output(rt, t230, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t230, all_m_vec, 224, 0, 32, 128);
    pto_task_submit(rt, t230);

    // Cross-tile: Q[7] x K/V[0]
    int32_t t231 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t231, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t231, all_k_rope, 0, 0, 32, 128);
    pto_task_add_output(rt, t231, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t231);

    int32_t t232 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t232, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t232, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t232, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t232, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t232, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t232, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t232, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t232);

    // Cross-tile: Q[7] x K/V[0]
    int32_t t233 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t233, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t233, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t233, all_v_tiles, 0, 0, 32, 128);
    pto_task_add_input(rt, t233, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t233, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t233);

    // Cross-tile: Q[7] x K/V[1]
    int32_t t234 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t234, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t234, all_k_rope, 32, 0, 32, 128);
    pto_task_add_output(rt, t234, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t234);

    int32_t t235 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t235, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t235, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t235, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t235, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t235, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t235, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t235, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t235);

    // Cross-tile: Q[7] x K/V[1]
    int32_t t236 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t236, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t236, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t236, all_v_tiles, 32, 0, 32, 128);
    pto_task_add_input(rt, t236, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t236, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t236);

    // Cross-tile: Q[7] x K/V[2]
    int32_t t237 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t237, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t237, all_k_rope, 64, 0, 32, 128);
    pto_task_add_output(rt, t237, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t237);

    int32_t t238 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t238, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t238, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t238, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t238, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t238, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t238, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t238, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t238);

    // Cross-tile: Q[7] x K/V[2]
    int32_t t239 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t239, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t239, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t239, all_v_tiles, 64, 0, 32, 128);
    pto_task_add_input(rt, t239, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t239, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t239);

    // Cross-tile: Q[7] x K/V[3]
    int32_t t240 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t240, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t240, all_k_rope, 96, 0, 32, 128);
    pto_task_add_output(rt, t240, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t240);

    int32_t t241 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t241, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t241, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t241, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t241, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t241, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t241, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t241, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t241);

    // Cross-tile: Q[7] x K/V[3]
    int32_t t242 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t242, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t242, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t242, all_v_tiles, 96, 0, 32, 128);
    pto_task_add_input(rt, t242, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t242, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t242);

    // Cross-tile: Q[7] x K/V[4]
    int32_t t243 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t243, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t243, all_k_rope, 128, 0, 32, 128);
    pto_task_add_output(rt, t243, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t243);

    int32_t t244 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t244, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t244, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t244, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t244, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t244, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t244, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t244, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t244);

    // Cross-tile: Q[7] x K/V[4]
    int32_t t245 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t245, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t245, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t245, all_v_tiles, 128, 0, 32, 128);
    pto_task_add_input(rt, t245, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t245, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t245);

    // Cross-tile: Q[7] x K/V[5]
    int32_t t246 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t246, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t246, all_k_rope, 160, 0, 32, 128);
    pto_task_add_output(rt, t246, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t246);

    int32_t t247 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t247, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t247, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t247, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t247, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t247, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t247, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t247, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t247);

    // Cross-tile: Q[7] x K/V[5]
    int32_t t248 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t248, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t248, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t248, all_v_tiles, 160, 0, 32, 128);
    pto_task_add_input(rt, t248, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t248, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t248);

    // Cross-tile: Q[7] x K/V[6]
    int32_t t249 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t249, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t249, all_k_rope, 192, 0, 32, 128);
    pto_task_add_output(rt, t249, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t249);

    int32_t t250 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t250, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t250, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t250, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t250, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t250, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t250, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t250, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t250);

    // Cross-tile: Q[7] x K/V[6]
    int32_t t251 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t251, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t251, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t251, all_v_tiles, 192, 0, 32, 128);
    pto_task_add_input(rt, t251, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t251, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t251);

    // Cross-tile: Q[7] x K/V[7]
    int32_t t252 = pto_task_alloc(rt, "flash_attn_score_block", NULL, 98304, 98304);
    pto_task_add_input(rt, t252, all_q_rope, 224, 0, 32, 128);
    pto_task_add_input(rt, t252, all_k_rope, 224, 0, 32, 128);
    pto_task_add_output(rt, t252, temp_scores, 224, 0, 32, 128);
    pto_task_submit(rt, t252);

    int32_t t253 = pto_task_alloc(rt, "flash_attn_softmax_update", NULL, 51456, 34048);
    pto_task_add_input(rt, t253, temp_scores, 224, 0, 32, 128);
    pto_task_add_input(rt, t253, all_m_vec, 224, 0, 32, 128);
    pto_task_add_input(rt, t253, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t253, all_m_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t253, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t253, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t253, temp_scale, 224, 0, 32, 128);
    pto_task_submit(rt, t253);

    // Cross-tile: Q[7] x K/V[7]
    int32_t t254 = pto_task_alloc(rt, "flash_attn_output_update", NULL, 180480, 147712);
    pto_task_add_input(rt, t254, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t254, temp_attn_weights, 224, 0, 32, 128);
    pto_task_add_input(rt, t254, all_v_tiles, 224, 0, 32, 128);
    pto_task_add_input(rt, t254, temp_scale, 224, 0, 32, 128);
    pto_task_add_output(rt, t254, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t254);

    int32_t t255 = pto_task_alloc(rt, "flash_attn_normalize", NULL, 65792, 65792);
    pto_task_add_input(rt, t255, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t255, all_l_vec, 224, 0, 32, 128);
    pto_task_add_output(rt, t255, all_attn_out, 224, 0, 32, 128);
    pto_task_submit(rt, t255);


    // ================================================================
    // PHASE 3: Post-Attention (depends on Phase 2 completion)
    // ================================================================
    // --- Tile 0 Post-Attention & MLP ---
    int32_t t256 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t256, all_attn_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t256, wo, 0, 0, 32, 128);
    pto_task_add_output(rt, t256, temp_norm, 0, 0, 32, 128);
    pto_task_submit(rt, t256);

    int32_t t257 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t257, temp_norm, 0, 0, 32, 128);
    pto_task_add_input(rt, t257, input, 0, 0, 32, 128);
    pto_task_add_output(rt, t257, all_hidden, 0, 0, 32, 128);
    pto_task_submit(rt, t257);

    int32_t t258 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t258, all_hidden, 0, 0, 32, 128);
    pto_task_add_input(rt, t258, mlp_norm_weights, 0, 0, 32, 128);
    pto_task_add_output(rt, t258, temp_norm, 0, 0, 32, 128);
    pto_task_submit(rt, t258);

    int32_t t259 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t259, temp_norm, 0, 0, 32, 128);
    pto_task_add_input(rt, t259, w_gate, 0, 0, 32, 128);
    pto_task_add_output(rt, t259, temp_gate, 0, 0, 32, 128);
    pto_task_submit(rt, t259);

    int32_t t260 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t260, temp_norm, 0, 0, 32, 128);
    pto_task_add_input(rt, t260, w_up, 0, 0, 32, 128);
    pto_task_add_output(rt, t260, temp_up, 0, 0, 32, 128);
    pto_task_submit(rt, t260);

    int32_t t261 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t261, temp_gate, 0, 0, 32, 128);
    pto_task_add_input(rt, t261, temp_up, 0, 0, 32, 128);
    pto_task_add_output(rt, t261, temp_swiglu, 0, 0, 32, 128);
    pto_task_submit(rt, t261);

    int32_t t262 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t262, temp_swiglu, 0, 0, 32, 128);
    pto_task_add_input(rt, t262, w_down, 0, 0, 32, 128);
    pto_task_add_output(rt, t262, temp_mlp_out, 0, 0, 32, 128);
    pto_task_submit(rt, t262);

    int32_t t263 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t263, temp_mlp_out, 0, 0, 32, 128);
    pto_task_add_input(rt, t263, all_hidden, 0, 0, 32, 128);
    pto_task_add_output(rt, t263, output, 0, 0, 32, 128);
    pto_task_submit(rt, t263);

    // --- Tile 1 Post-Attention & MLP ---
    int32_t t264 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t264, all_attn_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t264, wo, 32, 0, 32, 128);
    pto_task_add_output(rt, t264, temp_norm, 32, 0, 32, 128);
    pto_task_submit(rt, t264);

    int32_t t265 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t265, temp_norm, 32, 0, 32, 128);
    pto_task_add_input(rt, t265, input, 32, 0, 32, 128);
    pto_task_add_output(rt, t265, all_hidden, 32, 0, 32, 128);
    pto_task_submit(rt, t265);

    int32_t t266 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t266, all_hidden, 32, 0, 32, 128);
    pto_task_add_input(rt, t266, mlp_norm_weights, 32, 0, 32, 128);
    pto_task_add_output(rt, t266, temp_norm, 32, 0, 32, 128);
    pto_task_submit(rt, t266);

    int32_t t267 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t267, temp_norm, 32, 0, 32, 128);
    pto_task_add_input(rt, t267, w_gate, 32, 0, 32, 128);
    pto_task_add_output(rt, t267, temp_gate, 32, 0, 32, 128);
    pto_task_submit(rt, t267);

    int32_t t268 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t268, temp_norm, 32, 0, 32, 128);
    pto_task_add_input(rt, t268, w_up, 32, 0, 32, 128);
    pto_task_add_output(rt, t268, temp_up, 32, 0, 32, 128);
    pto_task_submit(rt, t268);

    int32_t t269 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t269, temp_gate, 32, 0, 32, 128);
    pto_task_add_input(rt, t269, temp_up, 32, 0, 32, 128);
    pto_task_add_output(rt, t269, temp_swiglu, 32, 0, 32, 128);
    pto_task_submit(rt, t269);

    int32_t t270 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t270, temp_swiglu, 32, 0, 32, 128);
    pto_task_add_input(rt, t270, w_down, 32, 0, 32, 128);
    pto_task_add_output(rt, t270, temp_mlp_out, 32, 0, 32, 128);
    pto_task_submit(rt, t270);

    int32_t t271 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t271, temp_mlp_out, 32, 0, 32, 128);
    pto_task_add_input(rt, t271, all_hidden, 32, 0, 32, 128);
    pto_task_add_output(rt, t271, output, 32, 0, 32, 128);
    pto_task_submit(rt, t271);

    // --- Tile 2 Post-Attention & MLP ---
    int32_t t272 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t272, all_attn_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t272, wo, 64, 0, 32, 128);
    pto_task_add_output(rt, t272, temp_norm, 64, 0, 32, 128);
    pto_task_submit(rt, t272);

    int32_t t273 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t273, temp_norm, 64, 0, 32, 128);
    pto_task_add_input(rt, t273, input, 64, 0, 32, 128);
    pto_task_add_output(rt, t273, all_hidden, 64, 0, 32, 128);
    pto_task_submit(rt, t273);

    int32_t t274 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t274, all_hidden, 64, 0, 32, 128);
    pto_task_add_input(rt, t274, mlp_norm_weights, 64, 0, 32, 128);
    pto_task_add_output(rt, t274, temp_norm, 64, 0, 32, 128);
    pto_task_submit(rt, t274);

    int32_t t275 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t275, temp_norm, 64, 0, 32, 128);
    pto_task_add_input(rt, t275, w_gate, 64, 0, 32, 128);
    pto_task_add_output(rt, t275, temp_gate, 64, 0, 32, 128);
    pto_task_submit(rt, t275);

    int32_t t276 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t276, temp_norm, 64, 0, 32, 128);
    pto_task_add_input(rt, t276, w_up, 64, 0, 32, 128);
    pto_task_add_output(rt, t276, temp_up, 64, 0, 32, 128);
    pto_task_submit(rt, t276);

    int32_t t277 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t277, temp_gate, 64, 0, 32, 128);
    pto_task_add_input(rt, t277, temp_up, 64, 0, 32, 128);
    pto_task_add_output(rt, t277, temp_swiglu, 64, 0, 32, 128);
    pto_task_submit(rt, t277);

    int32_t t278 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t278, temp_swiglu, 64, 0, 32, 128);
    pto_task_add_input(rt, t278, w_down, 64, 0, 32, 128);
    pto_task_add_output(rt, t278, temp_mlp_out, 64, 0, 32, 128);
    pto_task_submit(rt, t278);

    int32_t t279 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t279, temp_mlp_out, 64, 0, 32, 128);
    pto_task_add_input(rt, t279, all_hidden, 64, 0, 32, 128);
    pto_task_add_output(rt, t279, output, 64, 0, 32, 128);
    pto_task_submit(rt, t279);

    // --- Tile 3 Post-Attention & MLP ---
    int32_t t280 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t280, all_attn_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t280, wo, 96, 0, 32, 128);
    pto_task_add_output(rt, t280, temp_norm, 96, 0, 32, 128);
    pto_task_submit(rt, t280);

    int32_t t281 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t281, temp_norm, 96, 0, 32, 128);
    pto_task_add_input(rt, t281, input, 96, 0, 32, 128);
    pto_task_add_output(rt, t281, all_hidden, 96, 0, 32, 128);
    pto_task_submit(rt, t281);

    int32_t t282 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t282, all_hidden, 96, 0, 32, 128);
    pto_task_add_input(rt, t282, mlp_norm_weights, 96, 0, 32, 128);
    pto_task_add_output(rt, t282, temp_norm, 96, 0, 32, 128);
    pto_task_submit(rt, t282);

    int32_t t283 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t283, temp_norm, 96, 0, 32, 128);
    pto_task_add_input(rt, t283, w_gate, 96, 0, 32, 128);
    pto_task_add_output(rt, t283, temp_gate, 96, 0, 32, 128);
    pto_task_submit(rt, t283);

    int32_t t284 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t284, temp_norm, 96, 0, 32, 128);
    pto_task_add_input(rt, t284, w_up, 96, 0, 32, 128);
    pto_task_add_output(rt, t284, temp_up, 96, 0, 32, 128);
    pto_task_submit(rt, t284);

    int32_t t285 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t285, temp_gate, 96, 0, 32, 128);
    pto_task_add_input(rt, t285, temp_up, 96, 0, 32, 128);
    pto_task_add_output(rt, t285, temp_swiglu, 96, 0, 32, 128);
    pto_task_submit(rt, t285);

    int32_t t286 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t286, temp_swiglu, 96, 0, 32, 128);
    pto_task_add_input(rt, t286, w_down, 96, 0, 32, 128);
    pto_task_add_output(rt, t286, temp_mlp_out, 96, 0, 32, 128);
    pto_task_submit(rt, t286);

    int32_t t287 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t287, temp_mlp_out, 96, 0, 32, 128);
    pto_task_add_input(rt, t287, all_hidden, 96, 0, 32, 128);
    pto_task_add_output(rt, t287, output, 96, 0, 32, 128);
    pto_task_submit(rt, t287);

    // --- Tile 4 Post-Attention & MLP ---
    int32_t t288 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t288, all_attn_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t288, wo, 128, 0, 32, 128);
    pto_task_add_output(rt, t288, temp_norm, 128, 0, 32, 128);
    pto_task_submit(rt, t288);

    int32_t t289 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t289, temp_norm, 128, 0, 32, 128);
    pto_task_add_input(rt, t289, input, 128, 0, 32, 128);
    pto_task_add_output(rt, t289, all_hidden, 128, 0, 32, 128);
    pto_task_submit(rt, t289);

    int32_t t290 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t290, all_hidden, 128, 0, 32, 128);
    pto_task_add_input(rt, t290, mlp_norm_weights, 128, 0, 32, 128);
    pto_task_add_output(rt, t290, temp_norm, 128, 0, 32, 128);
    pto_task_submit(rt, t290);

    int32_t t291 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t291, temp_norm, 128, 0, 32, 128);
    pto_task_add_input(rt, t291, w_gate, 128, 0, 32, 128);
    pto_task_add_output(rt, t291, temp_gate, 128, 0, 32, 128);
    pto_task_submit(rt, t291);

    int32_t t292 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t292, temp_norm, 128, 0, 32, 128);
    pto_task_add_input(rt, t292, w_up, 128, 0, 32, 128);
    pto_task_add_output(rt, t292, temp_up, 128, 0, 32, 128);
    pto_task_submit(rt, t292);

    int32_t t293 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t293, temp_gate, 128, 0, 32, 128);
    pto_task_add_input(rt, t293, temp_up, 128, 0, 32, 128);
    pto_task_add_output(rt, t293, temp_swiglu, 128, 0, 32, 128);
    pto_task_submit(rt, t293);

    int32_t t294 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t294, temp_swiglu, 128, 0, 32, 128);
    pto_task_add_input(rt, t294, w_down, 128, 0, 32, 128);
    pto_task_add_output(rt, t294, temp_mlp_out, 128, 0, 32, 128);
    pto_task_submit(rt, t294);

    int32_t t295 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t295, temp_mlp_out, 128, 0, 32, 128);
    pto_task_add_input(rt, t295, all_hidden, 128, 0, 32, 128);
    pto_task_add_output(rt, t295, output, 128, 0, 32, 128);
    pto_task_submit(rt, t295);

    // --- Tile 5 Post-Attention & MLP ---
    int32_t t296 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t296, all_attn_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t296, wo, 160, 0, 32, 128);
    pto_task_add_output(rt, t296, temp_norm, 160, 0, 32, 128);
    pto_task_submit(rt, t296);

    int32_t t297 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t297, temp_norm, 160, 0, 32, 128);
    pto_task_add_input(rt, t297, input, 160, 0, 32, 128);
    pto_task_add_output(rt, t297, all_hidden, 160, 0, 32, 128);
    pto_task_submit(rt, t297);

    int32_t t298 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t298, all_hidden, 160, 0, 32, 128);
    pto_task_add_input(rt, t298, mlp_norm_weights, 160, 0, 32, 128);
    pto_task_add_output(rt, t298, temp_norm, 160, 0, 32, 128);
    pto_task_submit(rt, t298);

    int32_t t299 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t299, temp_norm, 160, 0, 32, 128);
    pto_task_add_input(rt, t299, w_gate, 160, 0, 32, 128);
    pto_task_add_output(rt, t299, temp_gate, 160, 0, 32, 128);
    pto_task_submit(rt, t299);

    int32_t t300 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t300, temp_norm, 160, 0, 32, 128);
    pto_task_add_input(rt, t300, w_up, 160, 0, 32, 128);
    pto_task_add_output(rt, t300, temp_up, 160, 0, 32, 128);
    pto_task_submit(rt, t300);

    int32_t t301 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t301, temp_gate, 160, 0, 32, 128);
    pto_task_add_input(rt, t301, temp_up, 160, 0, 32, 128);
    pto_task_add_output(rt, t301, temp_swiglu, 160, 0, 32, 128);
    pto_task_submit(rt, t301);

    int32_t t302 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t302, temp_swiglu, 160, 0, 32, 128);
    pto_task_add_input(rt, t302, w_down, 160, 0, 32, 128);
    pto_task_add_output(rt, t302, temp_mlp_out, 160, 0, 32, 128);
    pto_task_submit(rt, t302);

    int32_t t303 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t303, temp_mlp_out, 160, 0, 32, 128);
    pto_task_add_input(rt, t303, all_hidden, 160, 0, 32, 128);
    pto_task_add_output(rt, t303, output, 160, 0, 32, 128);
    pto_task_submit(rt, t303);

    // --- Tile 6 Post-Attention & MLP ---
    int32_t t304 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t304, all_attn_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t304, wo, 192, 0, 32, 128);
    pto_task_add_output(rt, t304, temp_norm, 192, 0, 32, 128);
    pto_task_submit(rt, t304);

    int32_t t305 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t305, temp_norm, 192, 0, 32, 128);
    pto_task_add_input(rt, t305, input, 192, 0, 32, 128);
    pto_task_add_output(rt, t305, all_hidden, 192, 0, 32, 128);
    pto_task_submit(rt, t305);

    int32_t t306 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t306, all_hidden, 192, 0, 32, 128);
    pto_task_add_input(rt, t306, mlp_norm_weights, 192, 0, 32, 128);
    pto_task_add_output(rt, t306, temp_norm, 192, 0, 32, 128);
    pto_task_submit(rt, t306);

    int32_t t307 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t307, temp_norm, 192, 0, 32, 128);
    pto_task_add_input(rt, t307, w_gate, 192, 0, 32, 128);
    pto_task_add_output(rt, t307, temp_gate, 192, 0, 32, 128);
    pto_task_submit(rt, t307);

    int32_t t308 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t308, temp_norm, 192, 0, 32, 128);
    pto_task_add_input(rt, t308, w_up, 192, 0, 32, 128);
    pto_task_add_output(rt, t308, temp_up, 192, 0, 32, 128);
    pto_task_submit(rt, t308);

    int32_t t309 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t309, temp_gate, 192, 0, 32, 128);
    pto_task_add_input(rt, t309, temp_up, 192, 0, 32, 128);
    pto_task_add_output(rt, t309, temp_swiglu, 192, 0, 32, 128);
    pto_task_submit(rt, t309);

    int32_t t310 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t310, temp_swiglu, 192, 0, 32, 128);
    pto_task_add_input(rt, t310, w_down, 192, 0, 32, 128);
    pto_task_add_output(rt, t310, temp_mlp_out, 192, 0, 32, 128);
    pto_task_submit(rt, t310);

    int32_t t311 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t311, temp_mlp_out, 192, 0, 32, 128);
    pto_task_add_input(rt, t311, all_hidden, 192, 0, 32, 128);
    pto_task_add_output(rt, t311, output, 192, 0, 32, 128);
    pto_task_submit(rt, t311);

    // --- Tile 7 Post-Attention & MLP ---
    int32_t t312 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t312, all_attn_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t312, wo, 224, 0, 32, 128);
    pto_task_add_output(rt, t312, temp_norm, 224, 0, 32, 128);
    pto_task_submit(rt, t312);

    int32_t t313 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t313, temp_norm, 224, 0, 32, 128);
    pto_task_add_input(rt, t313, input, 224, 0, 32, 128);
    pto_task_add_output(rt, t313, all_hidden, 224, 0, 32, 128);
    pto_task_submit(rt, t313);

    int32_t t314 = pto_task_alloc(rt, "rmsnorm_tile", NULL, 82304, 49408);
    pto_task_add_input(rt, t314, all_hidden, 224, 0, 32, 128);
    pto_task_add_input(rt, t314, mlp_norm_weights, 224, 0, 32, 128);
    pto_task_add_output(rt, t314, temp_norm, 224, 0, 32, 128);
    pto_task_submit(rt, t314);

    int32_t t315 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t315, temp_norm, 224, 0, 32, 128);
    pto_task_add_input(rt, t315, w_gate, 224, 0, 32, 128);
    pto_task_add_output(rt, t315, temp_gate, 224, 0, 32, 128);
    pto_task_submit(rt, t315);

    int32_t t316 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t316, temp_norm, 224, 0, 32, 128);
    pto_task_add_input(rt, t316, w_up, 224, 0, 32, 128);
    pto_task_add_output(rt, t316, temp_up, 224, 0, 32, 128);
    pto_task_submit(rt, t316);

    int32_t t317 = pto_task_alloc(rt, "swiglu_tile", NULL, 131072, 65536);
    pto_task_add_input(rt, t317, temp_gate, 224, 0, 32, 128);
    pto_task_add_input(rt, t317, temp_up, 224, 0, 32, 128);
    pto_task_add_output(rt, t317, temp_swiglu, 224, 0, 32, 128);
    pto_task_submit(rt, t317);

    int32_t t318 = pto_task_alloc(rt, "tile_matmul", NULL, 98304, 98304);
    pto_task_add_input(rt, t318, temp_swiglu, 224, 0, 32, 128);
    pto_task_add_input(rt, t318, w_down, 224, 0, 32, 128);
    pto_task_add_output(rt, t318, temp_mlp_out, 224, 0, 32, 128);
    pto_task_submit(rt, t318);

    int32_t t319 = pto_task_alloc(rt, "residual_add_tile", NULL, 49152, 49152);
    pto_task_add_input(rt, t319, temp_mlp_out, 224, 0, 32, 128);
    pto_task_add_input(rt, t319, all_hidden, 224, 0, 32, 128);
    pto_task_add_output(rt, t319, output, 224, 0, 32, 128);
    pto_task_submit(rt, t319);

}

int main(int argc, char** argv) {
    PTORuntime rt;
    pto_runtime_init(&rt);

    build_task_graph(&rt);

    printf("\n");
    pto_runtime_dump_stdout(&rt);
    pto_runtime_dump(&rt, "llama_layer_dynamic_task_graph.txt");

    pto_runtime_shutdown(&rt);
    return 0;
}