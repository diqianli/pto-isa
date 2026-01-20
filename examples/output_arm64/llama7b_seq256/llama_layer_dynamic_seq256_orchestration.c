/**
 * LLaMA 7B Layer Orchestration - seq_len=256
 * num_full_tiles: 32
 * tail_rows: 0
 */

#include "pto_runtime.h"
#include "pto_runtime.c"

void build_task_graph(PTORuntime* rt) {
    // Declare buffers
    float input[64];
    float output[64];
    float attn_norm_weights[64];
    float wq[64];
    float wk[64];
    float wv[64];
    float wo[64];
    float cos_cache[64];
    float sin_cache[64];
    float mlp_norm_weights[64];
    float w_gate[64];
    float w_up[64];
    float w_down[64];
    float temp_attn_out[64];
    float temp_residual1[64];
    float temp_mlp_out[64];
    float temp_norm[64];
    float temp_q[64];
    float temp_k[64];
    float temp_v[64];
    float temp_q_rope[64];
    float temp_k_rope[64];
    float temp_scores[64];
    float temp_attn_weights[64];
    float temp_gate[64];
    float temp_up[64];
    float temp_swiglu[64];

    // ========== Tile 0 ==========
    // Task 0: rmsnorm_tile
    int32_t t0 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t0, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t0, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t0, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t0);

    // Task 1: linear_tile
    int32_t t1 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t1, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t1, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t1, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t1);

    // Task 2: linear_tile
    int32_t t2 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t2, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t2, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t2, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t2);

    // Task 3: linear_tile
    int32_t t3 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t3, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t3, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t3, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t3);

    // Task 4: rope_tile
    int32_t t4 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t4, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t4, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t4, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t4, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t4);

    // Task 5: rope_tile
    int32_t t5 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t5, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t5, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t5, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t5, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t5);

    // Task 6: attention_score_tile
    int32_t t6 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t6, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t6, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t6, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t6);

    // Task 7: softmax_tile
    int32_t t7 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t7, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t7, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t7);

    // Task 8: attention_output_tile
    int32_t t8 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t8, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t8, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t8, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t8);

    // Task 9: linear_tile
    int32_t t9 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t9, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t9, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t9, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t9);

    // Task 10: residual_add_tile
    int32_t t10 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t10, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t10, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t10, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t10);

    // Task 11: rmsnorm_tile
    int32_t t11 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t11, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t11, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t11, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t11);

    // Task 12: linear_tile
    int32_t t12 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t12, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t12, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t12, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t12);

    // Task 13: linear_tile
    int32_t t13 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t13, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t13, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t13, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t13);

    // Task 14: swiglu_tile
    int32_t t14 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t14, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t14, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t14, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t14);

    // Task 15: linear_tile
    int32_t t15 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t15, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t15, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t15, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t15);

    // Task 16: residual_add_tile
    int32_t t16 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t16, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t16, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t16, output, 0, 0, 8, 8);
    pto_task_submit(rt, t16);

    // ========== Tile 1 ==========
    // Task 17: rmsnorm_tile
    int32_t t17 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t17, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t17, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t17, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t17);

    // Task 18: linear_tile
    int32_t t18 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t18, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t18, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t18, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t18);

    // Task 19: linear_tile
    int32_t t19 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t19, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t19, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t19, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t19);

    // Task 20: linear_tile
    int32_t t20 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t20, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t20, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t20, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t20);

    // Task 21: rope_tile
    int32_t t21 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t21, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t21, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t21, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t21, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t21);

    // Task 22: rope_tile
    int32_t t22 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t22, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t22, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t22, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t22, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t22);

    // Task 23: attention_score_tile
    int32_t t23 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t23, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t23, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t23, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t23);

    // Task 24: softmax_tile
    int32_t t24 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t24, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t24, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t24);

    // Task 25: attention_output_tile
    int32_t t25 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t25, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t25, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t25, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t25);

    // Task 26: linear_tile
    int32_t t26 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t26, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t26, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t26, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t26);

    // Task 27: residual_add_tile
    int32_t t27 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t27, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t27, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t27, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t27);

    // Task 28: rmsnorm_tile
    int32_t t28 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t28, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t28, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t28, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t28);

    // Task 29: linear_tile
    int32_t t29 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t29, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t29, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t29, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t29);

    // Task 30: linear_tile
    int32_t t30 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t30, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t30, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t30, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t30);

    // Task 31: swiglu_tile
    int32_t t31 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t31, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t31, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t31, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t31);

    // Task 32: linear_tile
    int32_t t32 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t32, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t32, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t32, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t32);

    // Task 33: residual_add_tile
    int32_t t33 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t33, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t33, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t33, output, 0, 0, 8, 8);
    pto_task_submit(rt, t33);

    // ========== Tile 2 ==========
    // Task 34: rmsnorm_tile
    int32_t t34 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t34, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t34, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t34, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t34);

    // Task 35: linear_tile
    int32_t t35 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t35, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t35, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t35, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t35);

    // Task 36: linear_tile
    int32_t t36 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t36, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t36, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t36, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t36);

    // Task 37: linear_tile
    int32_t t37 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t37, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t37, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t37, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t37);

    // Task 38: rope_tile
    int32_t t38 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t38, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t38, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t38, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t38, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t38);

    // Task 39: rope_tile
    int32_t t39 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t39, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t39, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t39, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t39, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t39);

    // Task 40: attention_score_tile
    int32_t t40 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t40, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t40, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t40, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t40);

    // Task 41: softmax_tile
    int32_t t41 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t41, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t41, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t41);

    // Task 42: attention_output_tile
    int32_t t42 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t42, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t42, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t42, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t42);

    // Task 43: linear_tile
    int32_t t43 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t43, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t43, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t43, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t43);

    // Task 44: residual_add_tile
    int32_t t44 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t44, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t44, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t44, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t44);

    // Task 45: rmsnorm_tile
    int32_t t45 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t45, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t45, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t45, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t45);

    // Task 46: linear_tile
    int32_t t46 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t46, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t46, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t46, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t46);

    // Task 47: linear_tile
    int32_t t47 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t47, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t47, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t47, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t47);

    // Task 48: swiglu_tile
    int32_t t48 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t48, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t48, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t48, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t48);

    // Task 49: linear_tile
    int32_t t49 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t49, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t49, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t49, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t49);

    // Task 50: residual_add_tile
    int32_t t50 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t50, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t50, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t50, output, 0, 0, 8, 8);
    pto_task_submit(rt, t50);

    // ========== Tile 3 ==========
    // Task 51: rmsnorm_tile
    int32_t t51 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t51, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t51, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t51, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t51);

    // Task 52: linear_tile
    int32_t t52 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t52, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t52, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t52, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t52);

    // Task 53: linear_tile
    int32_t t53 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t53, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t53, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t53, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t53);

    // Task 54: linear_tile
    int32_t t54 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t54, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t54, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t54, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t54);

    // Task 55: rope_tile
    int32_t t55 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t55, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t55, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t55, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t55, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t55);

    // Task 56: rope_tile
    int32_t t56 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t56, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t56, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t56, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t56, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t56);

    // Task 57: attention_score_tile
    int32_t t57 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t57, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t57, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t57, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t57);

    // Task 58: softmax_tile
    int32_t t58 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t58, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t58, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t58);

    // Task 59: attention_output_tile
    int32_t t59 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t59, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t59, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t59, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t59);

    // Task 60: linear_tile
    int32_t t60 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t60, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t60, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t60, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t60);

    // Task 61: residual_add_tile
    int32_t t61 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t61, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t61, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t61, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t61);

    // Task 62: rmsnorm_tile
    int32_t t62 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t62, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t62, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t62, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t62);

    // Task 63: linear_tile
    int32_t t63 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t63, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t63, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t63, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t63);

    // Task 64: linear_tile
    int32_t t64 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t64, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t64, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t64, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t64);

    // Task 65: swiglu_tile
    int32_t t65 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t65, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t65, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t65, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t65);

    // Task 66: linear_tile
    int32_t t66 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t66, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t66, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t66, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t66);

    // Task 67: residual_add_tile
    int32_t t67 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t67, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t67, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t67, output, 0, 0, 8, 8);
    pto_task_submit(rt, t67);

    // ========== Tile 4 ==========
    // Task 68: rmsnorm_tile
    int32_t t68 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t68, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t68, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t68, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t68);

    // Task 69: linear_tile
    int32_t t69 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t69, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t69, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t69, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t69);

    // Task 70: linear_tile
    int32_t t70 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t70, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t70, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t70, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t70);

    // Task 71: linear_tile
    int32_t t71 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t71, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t71, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t71, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t71);

    // Task 72: rope_tile
    int32_t t72 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t72, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t72, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t72, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t72, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t72);

    // Task 73: rope_tile
    int32_t t73 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t73, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t73, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t73, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t73, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t73);

    // Task 74: attention_score_tile
    int32_t t74 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t74, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t74, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t74, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t74);

    // Task 75: softmax_tile
    int32_t t75 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t75, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t75, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t75);

    // Task 76: attention_output_tile
    int32_t t76 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t76, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t76, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t76, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t76);

    // Task 77: linear_tile
    int32_t t77 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t77, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t77, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t77, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t77);

    // Task 78: residual_add_tile
    int32_t t78 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t78, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t78, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t78, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t78);

    // Task 79: rmsnorm_tile
    int32_t t79 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t79, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t79, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t79, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t79);

    // Task 80: linear_tile
    int32_t t80 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t80, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t80, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t80, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t80);

    // Task 81: linear_tile
    int32_t t81 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t81, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t81, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t81, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t81);

    // Task 82: swiglu_tile
    int32_t t82 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t82, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t82, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t82, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t82);

    // Task 83: linear_tile
    int32_t t83 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t83, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t83, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t83, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t83);

    // Task 84: residual_add_tile
    int32_t t84 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t84, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t84, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t84, output, 0, 0, 8, 8);
    pto_task_submit(rt, t84);

    // ========== Tile 5 ==========
    // Task 85: rmsnorm_tile
    int32_t t85 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t85, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t85, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t85, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t85);

    // Task 86: linear_tile
    int32_t t86 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t86, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t86, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t86, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t86);

    // Task 87: linear_tile
    int32_t t87 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t87, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t87, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t87, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t87);

    // Task 88: linear_tile
    int32_t t88 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t88, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t88, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t88, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t88);

    // Task 89: rope_tile
    int32_t t89 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t89, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t89, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t89, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t89, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t89);

    // Task 90: rope_tile
    int32_t t90 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t90, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t90, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t90, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t90, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t90);

    // Task 91: attention_score_tile
    int32_t t91 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t91, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t91, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t91, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t91);

    // Task 92: softmax_tile
    int32_t t92 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t92, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t92, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t92);

    // Task 93: attention_output_tile
    int32_t t93 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t93, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t93, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t93, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t93);

    // Task 94: linear_tile
    int32_t t94 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t94, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t94, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t94, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t94);

    // Task 95: residual_add_tile
    int32_t t95 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t95, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t95, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t95, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t95);

    // Task 96: rmsnorm_tile
    int32_t t96 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t96, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t96, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t96, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t96);

    // Task 97: linear_tile
    int32_t t97 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t97, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t97, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t97, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t97);

    // Task 98: linear_tile
    int32_t t98 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t98, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t98, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t98, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t98);

    // Task 99: swiglu_tile
    int32_t t99 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t99, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t99, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t99, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t99);

    // Task 100: linear_tile
    int32_t t100 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t100, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t100, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t100, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t100);

    // Task 101: residual_add_tile
    int32_t t101 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t101, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t101, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t101, output, 0, 0, 8, 8);
    pto_task_submit(rt, t101);

    // ========== Tile 6 ==========
    // Task 102: rmsnorm_tile
    int32_t t102 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t102, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t102, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t102, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t102);

    // Task 103: linear_tile
    int32_t t103 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t103, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t103, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t103, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t103);

    // Task 104: linear_tile
    int32_t t104 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t104, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t104, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t104, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t104);

    // Task 105: linear_tile
    int32_t t105 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t105, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t105, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t105, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t105);

    // Task 106: rope_tile
    int32_t t106 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t106, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t106, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t106, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t106, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t106);

    // Task 107: rope_tile
    int32_t t107 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t107, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t107, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t107, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t107, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t107);

    // Task 108: attention_score_tile
    int32_t t108 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t108, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t108, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t108, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t108);

    // Task 109: softmax_tile
    int32_t t109 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t109, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t109, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t109);

    // Task 110: attention_output_tile
    int32_t t110 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t110, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t110, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t110, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t110);

    // Task 111: linear_tile
    int32_t t111 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t111, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t111, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t111, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t111);

    // Task 112: residual_add_tile
    int32_t t112 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t112, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t112, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t112, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t112);

    // Task 113: rmsnorm_tile
    int32_t t113 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t113, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t113, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t113, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t113);

    // Task 114: linear_tile
    int32_t t114 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t114, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t114, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t114, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t114);

    // Task 115: linear_tile
    int32_t t115 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t115, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t115, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t115, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t115);

    // Task 116: swiglu_tile
    int32_t t116 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t116, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t116, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t116, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t116);

    // Task 117: linear_tile
    int32_t t117 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t117, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t117, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t117, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t117);

    // Task 118: residual_add_tile
    int32_t t118 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t118, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t118, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t118, output, 0, 0, 8, 8);
    pto_task_submit(rt, t118);

    // ========== Tile 7 ==========
    // Task 119: rmsnorm_tile
    int32_t t119 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t119, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t119, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t119, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t119);

    // Task 120: linear_tile
    int32_t t120 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t120, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t120, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t120, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t120);

    // Task 121: linear_tile
    int32_t t121 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t121, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t121, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t121, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t121);

    // Task 122: linear_tile
    int32_t t122 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t122, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t122, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t122, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t122);

    // Task 123: rope_tile
    int32_t t123 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t123, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t123, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t123, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t123, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t123);

    // Task 124: rope_tile
    int32_t t124 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t124, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t124, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t124, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t124, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t124);

    // Task 125: attention_score_tile
    int32_t t125 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t125, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t125, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t125, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t125);

    // Task 126: softmax_tile
    int32_t t126 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t126, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t126, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t126);

    // Task 127: attention_output_tile
    int32_t t127 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t127, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t127, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t127, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t127);

    // Task 128: linear_tile
    int32_t t128 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t128, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t128, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t128, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t128);

    // Task 129: residual_add_tile
    int32_t t129 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t129, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t129, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t129, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t129);

    // Task 130: rmsnorm_tile
    int32_t t130 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t130, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t130, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t130, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t130);

    // Task 131: linear_tile
    int32_t t131 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t131, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t131, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t131, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t131);

    // Task 132: linear_tile
    int32_t t132 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t132, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t132, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t132, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t132);

    // Task 133: swiglu_tile
    int32_t t133 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t133, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t133, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t133, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t133);

    // Task 134: linear_tile
    int32_t t134 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t134, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t134, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t134, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t134);

    // Task 135: residual_add_tile
    int32_t t135 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t135, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t135, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t135, output, 0, 0, 8, 8);
    pto_task_submit(rt, t135);

    // ========== Tile 8 ==========
    // Task 136: rmsnorm_tile
    int32_t t136 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t136, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t136, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t136, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t136);

    // Task 137: linear_tile
    int32_t t137 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t137, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t137, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t137, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t137);

    // Task 138: linear_tile
    int32_t t138 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t138, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t138, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t138, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t138);

    // Task 139: linear_tile
    int32_t t139 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t139, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t139, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t139, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t139);

    // Task 140: rope_tile
    int32_t t140 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t140, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t140, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t140, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t140, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t140);

    // Task 141: rope_tile
    int32_t t141 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t141, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t141, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t141, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t141, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t141);

    // Task 142: attention_score_tile
    int32_t t142 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t142, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t142, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t142, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t142);

    // Task 143: softmax_tile
    int32_t t143 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t143, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t143, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t143);

    // Task 144: attention_output_tile
    int32_t t144 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t144, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t144, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t144, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t144);

    // Task 145: linear_tile
    int32_t t145 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t145, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t145, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t145, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t145);

    // Task 146: residual_add_tile
    int32_t t146 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t146, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t146, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t146, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t146);

    // Task 147: rmsnorm_tile
    int32_t t147 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t147, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t147, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t147, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t147);

    // Task 148: linear_tile
    int32_t t148 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t148, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t148, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t148, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t148);

    // Task 149: linear_tile
    int32_t t149 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t149, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t149, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t149, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t149);

    // Task 150: swiglu_tile
    int32_t t150 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t150, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t150, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t150, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t150);

    // Task 151: linear_tile
    int32_t t151 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t151, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t151, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t151, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t151);

    // Task 152: residual_add_tile
    int32_t t152 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t152, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t152, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t152, output, 0, 0, 8, 8);
    pto_task_submit(rt, t152);

    // ========== Tile 9 ==========
    // Task 153: rmsnorm_tile
    int32_t t153 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t153, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t153, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t153, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t153);

    // Task 154: linear_tile
    int32_t t154 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t154, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t154, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t154, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t154);

    // Task 155: linear_tile
    int32_t t155 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t155, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t155, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t155, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t155);

    // Task 156: linear_tile
    int32_t t156 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t156, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t156, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t156, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t156);

    // Task 157: rope_tile
    int32_t t157 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t157, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t157, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t157, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t157, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t157);

    // Task 158: rope_tile
    int32_t t158 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t158, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t158, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t158, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t158, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t158);

    // Task 159: attention_score_tile
    int32_t t159 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t159, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t159, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t159, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t159);

    // Task 160: softmax_tile
    int32_t t160 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t160, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t160, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t160);

    // Task 161: attention_output_tile
    int32_t t161 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t161, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t161, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t161, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t161);

    // Task 162: linear_tile
    int32_t t162 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t162, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t162, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t162, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t162);

    // Task 163: residual_add_tile
    int32_t t163 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t163, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t163, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t163, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t163);

    // Task 164: rmsnorm_tile
    int32_t t164 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t164, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t164, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t164, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t164);

    // Task 165: linear_tile
    int32_t t165 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t165, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t165, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t165, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t165);

    // Task 166: linear_tile
    int32_t t166 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t166, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t166, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t166, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t166);

    // Task 167: swiglu_tile
    int32_t t167 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t167, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t167, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t167, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t167);

    // Task 168: linear_tile
    int32_t t168 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t168, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t168, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t168, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t168);

    // Task 169: residual_add_tile
    int32_t t169 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t169, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t169, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t169, output, 0, 0, 8, 8);
    pto_task_submit(rt, t169);

    // ========== Tile 10 ==========
    // Task 170: rmsnorm_tile
    int32_t t170 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t170, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t170, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t170, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t170);

    // Task 171: linear_tile
    int32_t t171 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t171, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t171, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t171, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t171);

    // Task 172: linear_tile
    int32_t t172 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t172, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t172, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t172, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t172);

    // Task 173: linear_tile
    int32_t t173 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t173, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t173, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t173, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t173);

    // Task 174: rope_tile
    int32_t t174 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t174, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t174, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t174, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t174, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t174);

    // Task 175: rope_tile
    int32_t t175 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t175, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t175, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t175, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t175, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t175);

    // Task 176: attention_score_tile
    int32_t t176 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t176, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t176, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t176, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t176);

    // Task 177: softmax_tile
    int32_t t177 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t177, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t177, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t177);

    // Task 178: attention_output_tile
    int32_t t178 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t178, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t178, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t178, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t178);

    // Task 179: linear_tile
    int32_t t179 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t179, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t179, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t179, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t179);

    // Task 180: residual_add_tile
    int32_t t180 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t180, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t180, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t180, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t180);

    // Task 181: rmsnorm_tile
    int32_t t181 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t181, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t181, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t181, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t181);

    // Task 182: linear_tile
    int32_t t182 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t182, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t182, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t182, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t182);

    // Task 183: linear_tile
    int32_t t183 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t183, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t183, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t183, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t183);

    // Task 184: swiglu_tile
    int32_t t184 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t184, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t184, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t184, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t184);

    // Task 185: linear_tile
    int32_t t185 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t185, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t185, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t185, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t185);

    // Task 186: residual_add_tile
    int32_t t186 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t186, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t186, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t186, output, 0, 0, 8, 8);
    pto_task_submit(rt, t186);

    // ========== Tile 11 ==========
    // Task 187: rmsnorm_tile
    int32_t t187 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t187, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t187, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t187, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t187);

    // Task 188: linear_tile
    int32_t t188 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t188, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t188, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t188, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t188);

    // Task 189: linear_tile
    int32_t t189 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t189, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t189, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t189, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t189);

    // Task 190: linear_tile
    int32_t t190 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t190, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t190, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t190, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t190);

    // Task 191: rope_tile
    int32_t t191 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t191, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t191, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t191, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t191, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t191);

    // Task 192: rope_tile
    int32_t t192 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t192, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t192, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t192, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t192, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t192);

    // Task 193: attention_score_tile
    int32_t t193 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t193, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t193, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t193, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t193);

    // Task 194: softmax_tile
    int32_t t194 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t194, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t194, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t194);

    // Task 195: attention_output_tile
    int32_t t195 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t195, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t195, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t195, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t195);

    // Task 196: linear_tile
    int32_t t196 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t196, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t196, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t196, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t196);

    // Task 197: residual_add_tile
    int32_t t197 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t197, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t197, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t197, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t197);

    // Task 198: rmsnorm_tile
    int32_t t198 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t198, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t198, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t198, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t198);

    // Task 199: linear_tile
    int32_t t199 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t199, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t199, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t199, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t199);

    // Task 200: linear_tile
    int32_t t200 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t200, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t200, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t200, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t200);

    // Task 201: swiglu_tile
    int32_t t201 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t201, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t201, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t201, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t201);

    // Task 202: linear_tile
    int32_t t202 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t202, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t202, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t202, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t202);

    // Task 203: residual_add_tile
    int32_t t203 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t203, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t203, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t203, output, 0, 0, 8, 8);
    pto_task_submit(rt, t203);

    // ========== Tile 12 ==========
    // Task 204: rmsnorm_tile
    int32_t t204 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t204, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t204, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t204, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t204);

    // Task 205: linear_tile
    int32_t t205 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t205, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t205, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t205, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t205);

    // Task 206: linear_tile
    int32_t t206 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t206, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t206, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t206, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t206);

    // Task 207: linear_tile
    int32_t t207 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t207, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t207, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t207, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t207);

    // Task 208: rope_tile
    int32_t t208 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t208, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t208, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t208, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t208, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t208);

    // Task 209: rope_tile
    int32_t t209 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t209, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t209, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t209, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t209, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t209);

    // Task 210: attention_score_tile
    int32_t t210 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t210, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t210, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t210, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t210);

    // Task 211: softmax_tile
    int32_t t211 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t211, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t211, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t211);

    // Task 212: attention_output_tile
    int32_t t212 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t212, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t212, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t212, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t212);

    // Task 213: linear_tile
    int32_t t213 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t213, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t213, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t213, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t213);

    // Task 214: residual_add_tile
    int32_t t214 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t214, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t214, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t214, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t214);

    // Task 215: rmsnorm_tile
    int32_t t215 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t215, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t215, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t215, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t215);

    // Task 216: linear_tile
    int32_t t216 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t216, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t216, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t216, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t216);

    // Task 217: linear_tile
    int32_t t217 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t217, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t217, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t217, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t217);

    // Task 218: swiglu_tile
    int32_t t218 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t218, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t218, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t218, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t218);

    // Task 219: linear_tile
    int32_t t219 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t219, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t219, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t219, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t219);

    // Task 220: residual_add_tile
    int32_t t220 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t220, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t220, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t220, output, 0, 0, 8, 8);
    pto_task_submit(rt, t220);

    // ========== Tile 13 ==========
    // Task 221: rmsnorm_tile
    int32_t t221 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t221, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t221, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t221, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t221);

    // Task 222: linear_tile
    int32_t t222 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t222, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t222, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t222, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t222);

    // Task 223: linear_tile
    int32_t t223 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t223, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t223, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t223, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t223);

    // Task 224: linear_tile
    int32_t t224 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t224, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t224, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t224, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t224);

    // Task 225: rope_tile
    int32_t t225 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t225, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t225, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t225, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t225, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t225);

    // Task 226: rope_tile
    int32_t t226 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t226, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t226, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t226, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t226, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t226);

    // Task 227: attention_score_tile
    int32_t t227 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t227, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t227, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t227, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t227);

    // Task 228: softmax_tile
    int32_t t228 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t228, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t228, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t228);

    // Task 229: attention_output_tile
    int32_t t229 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t229, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t229, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t229, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t229);

    // Task 230: linear_tile
    int32_t t230 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t230, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t230, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t230, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t230);

    // Task 231: residual_add_tile
    int32_t t231 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t231, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t231, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t231, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t231);

    // Task 232: rmsnorm_tile
    int32_t t232 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t232, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t232, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t232, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t232);

    // Task 233: linear_tile
    int32_t t233 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t233, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t233, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t233, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t233);

    // Task 234: linear_tile
    int32_t t234 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t234, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t234, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t234, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t234);

    // Task 235: swiglu_tile
    int32_t t235 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t235, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t235, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t235, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t235);

    // Task 236: linear_tile
    int32_t t236 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t236, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t236, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t236, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t236);

    // Task 237: residual_add_tile
    int32_t t237 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t237, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t237, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t237, output, 0, 0, 8, 8);
    pto_task_submit(rt, t237);

    // ========== Tile 14 ==========
    // Task 238: rmsnorm_tile
    int32_t t238 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t238, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t238, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t238, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t238);

    // Task 239: linear_tile
    int32_t t239 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t239, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t239, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t239, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t239);

    // Task 240: linear_tile
    int32_t t240 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t240, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t240, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t240, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t240);

    // Task 241: linear_tile
    int32_t t241 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t241, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t241, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t241, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t241);

    // Task 242: rope_tile
    int32_t t242 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t242, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t242, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t242, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t242, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t242);

    // Task 243: rope_tile
    int32_t t243 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t243, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t243, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t243, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t243, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t243);

    // Task 244: attention_score_tile
    int32_t t244 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t244, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t244, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t244, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t244);

    // Task 245: softmax_tile
    int32_t t245 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t245, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t245, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t245);

    // Task 246: attention_output_tile
    int32_t t246 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t246, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t246, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t246, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t246);

    // Task 247: linear_tile
    int32_t t247 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t247, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t247, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t247, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t247);

    // Task 248: residual_add_tile
    int32_t t248 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t248, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t248, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t248, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t248);

    // Task 249: rmsnorm_tile
    int32_t t249 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t249, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t249, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t249, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t249);

    // Task 250: linear_tile
    int32_t t250 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t250, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t250, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t250, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t250);

    // Task 251: linear_tile
    int32_t t251 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t251, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t251, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t251, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t251);

    // Task 252: swiglu_tile
    int32_t t252 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t252, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t252, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t252, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t252);

    // Task 253: linear_tile
    int32_t t253 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t253, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t253, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t253, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t253);

    // Task 254: residual_add_tile
    int32_t t254 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t254, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t254, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t254, output, 0, 0, 8, 8);
    pto_task_submit(rt, t254);

    // ========== Tile 15 ==========
    // Task 255: rmsnorm_tile
    int32_t t255 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t255, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t255, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t255, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t255);

    // Task 256: linear_tile
    int32_t t256 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t256, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t256, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t256, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t256);

    // Task 257: linear_tile
    int32_t t257 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t257, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t257, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t257, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t257);

    // Task 258: linear_tile
    int32_t t258 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t258, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t258, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t258, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t258);

    // Task 259: rope_tile
    int32_t t259 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t259, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t259, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t259, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t259, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t259);

    // Task 260: rope_tile
    int32_t t260 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t260, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t260, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t260, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t260, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t260);

    // Task 261: attention_score_tile
    int32_t t261 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t261, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t261, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t261, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t261);

    // Task 262: softmax_tile
    int32_t t262 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t262, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t262, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t262);

    // Task 263: attention_output_tile
    int32_t t263 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t263, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t263, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t263, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t263);

    // Task 264: linear_tile
    int32_t t264 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t264, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t264, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t264, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t264);

    // Task 265: residual_add_tile
    int32_t t265 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t265, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t265, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t265, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t265);

    // Task 266: rmsnorm_tile
    int32_t t266 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t266, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t266, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t266, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t266);

    // Task 267: linear_tile
    int32_t t267 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t267, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t267, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t267, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t267);

    // Task 268: linear_tile
    int32_t t268 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t268, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t268, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t268, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t268);

    // Task 269: swiglu_tile
    int32_t t269 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t269, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t269, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t269, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t269);

    // Task 270: linear_tile
    int32_t t270 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t270, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t270, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t270, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t270);

    // Task 271: residual_add_tile
    int32_t t271 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t271, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t271, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t271, output, 0, 0, 8, 8);
    pto_task_submit(rt, t271);

    // ========== Tile 16 ==========
    // Task 272: rmsnorm_tile
    int32_t t272 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t272, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t272, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t272, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t272);

    // Task 273: linear_tile
    int32_t t273 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t273, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t273, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t273, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t273);

    // Task 274: linear_tile
    int32_t t274 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t274, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t274, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t274, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t274);

    // Task 275: linear_tile
    int32_t t275 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t275, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t275, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t275, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t275);

    // Task 276: rope_tile
    int32_t t276 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t276, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t276, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t276, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t276, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t276);

    // Task 277: rope_tile
    int32_t t277 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t277, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t277, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t277, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t277, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t277);

    // Task 278: attention_score_tile
    int32_t t278 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t278, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t278, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t278, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t278);

    // Task 279: softmax_tile
    int32_t t279 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t279, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t279, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t279);

    // Task 280: attention_output_tile
    int32_t t280 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t280, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t280, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t280, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t280);

    // Task 281: linear_tile
    int32_t t281 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t281, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t281, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t281, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t281);

    // Task 282: residual_add_tile
    int32_t t282 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t282, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t282, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t282, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t282);

    // Task 283: rmsnorm_tile
    int32_t t283 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t283, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t283, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t283, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t283);

    // Task 284: linear_tile
    int32_t t284 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t284, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t284, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t284, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t284);

    // Task 285: linear_tile
    int32_t t285 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t285, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t285, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t285, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t285);

    // Task 286: swiglu_tile
    int32_t t286 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t286, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t286, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t286, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t286);

    // Task 287: linear_tile
    int32_t t287 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t287, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t287, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t287, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t287);

    // Task 288: residual_add_tile
    int32_t t288 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t288, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t288, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t288, output, 0, 0, 8, 8);
    pto_task_submit(rt, t288);

    // ========== Tile 17 ==========
    // Task 289: rmsnorm_tile
    int32_t t289 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t289, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t289, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t289, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t289);

    // Task 290: linear_tile
    int32_t t290 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t290, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t290, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t290, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t290);

    // Task 291: linear_tile
    int32_t t291 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t291, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t291, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t291, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t291);

    // Task 292: linear_tile
    int32_t t292 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t292, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t292, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t292, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t292);

    // Task 293: rope_tile
    int32_t t293 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t293, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t293, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t293, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t293, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t293);

    // Task 294: rope_tile
    int32_t t294 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t294, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t294, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t294, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t294, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t294);

    // Task 295: attention_score_tile
    int32_t t295 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t295, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t295, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t295, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t295);

    // Task 296: softmax_tile
    int32_t t296 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t296, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t296, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t296);

    // Task 297: attention_output_tile
    int32_t t297 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t297, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t297, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t297, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t297);

    // Task 298: linear_tile
    int32_t t298 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t298, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t298, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t298, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t298);

    // Task 299: residual_add_tile
    int32_t t299 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t299, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t299, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t299, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t299);

    // Task 300: rmsnorm_tile
    int32_t t300 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t300, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t300, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t300, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t300);

    // Task 301: linear_tile
    int32_t t301 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t301, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t301, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t301, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t301);

    // Task 302: linear_tile
    int32_t t302 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t302, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t302, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t302, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t302);

    // Task 303: swiglu_tile
    int32_t t303 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t303, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t303, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t303, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t303);

    // Task 304: linear_tile
    int32_t t304 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t304, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t304, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t304, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t304);

    // Task 305: residual_add_tile
    int32_t t305 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t305, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t305, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t305, output, 0, 0, 8, 8);
    pto_task_submit(rt, t305);

    // ========== Tile 18 ==========
    // Task 306: rmsnorm_tile
    int32_t t306 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t306, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t306, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t306, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t306);

    // Task 307: linear_tile
    int32_t t307 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t307, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t307, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t307, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t307);

    // Task 308: linear_tile
    int32_t t308 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t308, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t308, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t308, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t308);

    // Task 309: linear_tile
    int32_t t309 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t309, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t309, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t309, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t309);

    // Task 310: rope_tile
    int32_t t310 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t310, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t310, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t310, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t310, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t310);

    // Task 311: rope_tile
    int32_t t311 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t311, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t311, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t311, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t311, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t311);

    // Task 312: attention_score_tile
    int32_t t312 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t312, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t312, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t312, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t312);

    // Task 313: softmax_tile
    int32_t t313 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t313, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t313, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t313);

    // Task 314: attention_output_tile
    int32_t t314 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t314, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t314, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t314, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t314);

    // Task 315: linear_tile
    int32_t t315 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t315, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t315, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t315, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t315);

    // Task 316: residual_add_tile
    int32_t t316 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t316, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t316, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t316, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t316);

    // Task 317: rmsnorm_tile
    int32_t t317 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t317, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t317, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t317, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t317);

    // Task 318: linear_tile
    int32_t t318 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t318, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t318, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t318, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t318);

    // Task 319: linear_tile
    int32_t t319 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t319, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t319, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t319, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t319);

    // Task 320: swiglu_tile
    int32_t t320 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t320, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t320, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t320, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t320);

    // Task 321: linear_tile
    int32_t t321 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t321, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t321, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t321, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t321);

    // Task 322: residual_add_tile
    int32_t t322 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t322, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t322, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t322, output, 0, 0, 8, 8);
    pto_task_submit(rt, t322);

    // ========== Tile 19 ==========
    // Task 323: rmsnorm_tile
    int32_t t323 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t323, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t323, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t323, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t323);

    // Task 324: linear_tile
    int32_t t324 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t324, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t324, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t324, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t324);

    // Task 325: linear_tile
    int32_t t325 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t325, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t325, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t325, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t325);

    // Task 326: linear_tile
    int32_t t326 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t326, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t326, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t326, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t326);

    // Task 327: rope_tile
    int32_t t327 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t327, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t327, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t327, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t327, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t327);

    // Task 328: rope_tile
    int32_t t328 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t328, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t328, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t328, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t328, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t328);

    // Task 329: attention_score_tile
    int32_t t329 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t329, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t329, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t329, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t329);

    // Task 330: softmax_tile
    int32_t t330 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t330, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t330, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t330);

    // Task 331: attention_output_tile
    int32_t t331 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t331, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t331, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t331, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t331);

    // Task 332: linear_tile
    int32_t t332 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t332, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t332, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t332, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t332);

    // Task 333: residual_add_tile
    int32_t t333 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t333, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t333, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t333, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t333);

    // Task 334: rmsnorm_tile
    int32_t t334 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t334, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t334, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t334, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t334);

    // Task 335: linear_tile
    int32_t t335 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t335, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t335, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t335, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t335);

    // Task 336: linear_tile
    int32_t t336 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t336, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t336, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t336, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t336);

    // Task 337: swiglu_tile
    int32_t t337 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t337, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t337, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t337, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t337);

    // Task 338: linear_tile
    int32_t t338 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t338, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t338, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t338, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t338);

    // Task 339: residual_add_tile
    int32_t t339 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t339, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t339, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t339, output, 0, 0, 8, 8);
    pto_task_submit(rt, t339);

    // ========== Tile 20 ==========
    // Task 340: rmsnorm_tile
    int32_t t340 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t340, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t340, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t340, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t340);

    // Task 341: linear_tile
    int32_t t341 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t341, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t341, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t341, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t341);

    // Task 342: linear_tile
    int32_t t342 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t342, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t342, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t342, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t342);

    // Task 343: linear_tile
    int32_t t343 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t343, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t343, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t343, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t343);

    // Task 344: rope_tile
    int32_t t344 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t344, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t344, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t344, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t344, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t344);

    // Task 345: rope_tile
    int32_t t345 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t345, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t345, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t345, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t345, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t345);

    // Task 346: attention_score_tile
    int32_t t346 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t346, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t346, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t346, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t346);

    // Task 347: softmax_tile
    int32_t t347 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t347, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t347, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t347);

    // Task 348: attention_output_tile
    int32_t t348 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t348, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t348, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t348, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t348);

    // Task 349: linear_tile
    int32_t t349 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t349, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t349, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t349, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t349);

    // Task 350: residual_add_tile
    int32_t t350 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t350, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t350, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t350, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t350);

    // Task 351: rmsnorm_tile
    int32_t t351 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t351, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t351, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t351, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t351);

    // Task 352: linear_tile
    int32_t t352 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t352, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t352, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t352, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t352);

    // Task 353: linear_tile
    int32_t t353 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t353, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t353, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t353, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t353);

    // Task 354: swiglu_tile
    int32_t t354 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t354, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t354, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t354, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t354);

    // Task 355: linear_tile
    int32_t t355 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t355, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t355, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t355, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t355);

    // Task 356: residual_add_tile
    int32_t t356 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t356, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t356, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t356, output, 0, 0, 8, 8);
    pto_task_submit(rt, t356);

    // ========== Tile 21 ==========
    // Task 357: rmsnorm_tile
    int32_t t357 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t357, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t357, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t357, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t357);

    // Task 358: linear_tile
    int32_t t358 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t358, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t358, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t358, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t358);

    // Task 359: linear_tile
    int32_t t359 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t359, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t359, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t359, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t359);

    // Task 360: linear_tile
    int32_t t360 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t360, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t360, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t360, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t360);

    // Task 361: rope_tile
    int32_t t361 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t361, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t361, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t361, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t361, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t361);

    // Task 362: rope_tile
    int32_t t362 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t362, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t362, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t362, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t362, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t362);

    // Task 363: attention_score_tile
    int32_t t363 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t363, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t363, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t363, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t363);

    // Task 364: softmax_tile
    int32_t t364 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t364, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t364, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t364);

    // Task 365: attention_output_tile
    int32_t t365 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t365, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t365, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t365, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t365);

    // Task 366: linear_tile
    int32_t t366 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t366, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t366, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t366, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t366);

    // Task 367: residual_add_tile
    int32_t t367 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t367, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t367, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t367, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t367);

    // Task 368: rmsnorm_tile
    int32_t t368 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t368, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t368, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t368, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t368);

    // Task 369: linear_tile
    int32_t t369 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t369, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t369, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t369, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t369);

    // Task 370: linear_tile
    int32_t t370 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t370, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t370, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t370, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t370);

    // Task 371: swiglu_tile
    int32_t t371 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t371, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t371, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t371, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t371);

    // Task 372: linear_tile
    int32_t t372 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t372, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t372, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t372, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t372);

    // Task 373: residual_add_tile
    int32_t t373 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t373, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t373, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t373, output, 0, 0, 8, 8);
    pto_task_submit(rt, t373);

    // ========== Tile 22 ==========
    // Task 374: rmsnorm_tile
    int32_t t374 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t374, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t374, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t374, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t374);

    // Task 375: linear_tile
    int32_t t375 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t375, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t375, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t375, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t375);

    // Task 376: linear_tile
    int32_t t376 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t376, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t376, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t376, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t376);

    // Task 377: linear_tile
    int32_t t377 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t377, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t377, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t377, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t377);

    // Task 378: rope_tile
    int32_t t378 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t378, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t378, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t378, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t378, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t378);

    // Task 379: rope_tile
    int32_t t379 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t379, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t379, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t379, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t379, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t379);

    // Task 380: attention_score_tile
    int32_t t380 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t380, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t380, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t380, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t380);

    // Task 381: softmax_tile
    int32_t t381 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t381, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t381, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t381);

    // Task 382: attention_output_tile
    int32_t t382 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t382, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t382, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t382, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t382);

    // Task 383: linear_tile
    int32_t t383 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t383, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t383, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t383, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t383);

    // Task 384: residual_add_tile
    int32_t t384 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t384, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t384, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t384, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t384);

    // Task 385: rmsnorm_tile
    int32_t t385 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t385, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t385, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t385, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t385);

    // Task 386: linear_tile
    int32_t t386 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t386, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t386, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t386, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t386);

    // Task 387: linear_tile
    int32_t t387 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t387, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t387, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t387, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t387);

    // Task 388: swiglu_tile
    int32_t t388 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t388, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t388, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t388, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t388);

    // Task 389: linear_tile
    int32_t t389 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t389, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t389, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t389, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t389);

    // Task 390: residual_add_tile
    int32_t t390 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t390, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t390, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t390, output, 0, 0, 8, 8);
    pto_task_submit(rt, t390);

    // ========== Tile 23 ==========
    // Task 391: rmsnorm_tile
    int32_t t391 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t391, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t391, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t391, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t391);

    // Task 392: linear_tile
    int32_t t392 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t392, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t392, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t392, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t392);

    // Task 393: linear_tile
    int32_t t393 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t393, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t393, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t393, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t393);

    // Task 394: linear_tile
    int32_t t394 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t394, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t394, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t394, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t394);

    // Task 395: rope_tile
    int32_t t395 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t395, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t395, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t395, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t395, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t395);

    // Task 396: rope_tile
    int32_t t396 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t396, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t396, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t396, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t396, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t396);

    // Task 397: attention_score_tile
    int32_t t397 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t397, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t397, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t397, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t397);

    // Task 398: softmax_tile
    int32_t t398 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t398, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t398, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t398);

    // Task 399: attention_output_tile
    int32_t t399 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t399, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t399, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t399, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t399);

    // Task 400: linear_tile
    int32_t t400 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t400, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t400, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t400, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t400);

    // Task 401: residual_add_tile
    int32_t t401 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t401, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t401, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t401, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t401);

    // Task 402: rmsnorm_tile
    int32_t t402 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t402, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t402, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t402, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t402);

    // Task 403: linear_tile
    int32_t t403 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t403, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t403, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t403, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t403);

    // Task 404: linear_tile
    int32_t t404 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t404, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t404, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t404, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t404);

    // Task 405: swiglu_tile
    int32_t t405 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t405, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t405, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t405, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t405);

    // Task 406: linear_tile
    int32_t t406 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t406, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t406, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t406, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t406);

    // Task 407: residual_add_tile
    int32_t t407 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t407, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t407, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t407, output, 0, 0, 8, 8);
    pto_task_submit(rt, t407);

    // ========== Tile 24 ==========
    // Task 408: rmsnorm_tile
    int32_t t408 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t408, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t408, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t408, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t408);

    // Task 409: linear_tile
    int32_t t409 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t409, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t409, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t409, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t409);

    // Task 410: linear_tile
    int32_t t410 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t410, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t410, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t410, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t410);

    // Task 411: linear_tile
    int32_t t411 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t411, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t411, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t411, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t411);

    // Task 412: rope_tile
    int32_t t412 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t412, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t412, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t412, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t412, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t412);

    // Task 413: rope_tile
    int32_t t413 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t413, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t413, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t413, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t413, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t413);

    // Task 414: attention_score_tile
    int32_t t414 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t414, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t414, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t414, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t414);

    // Task 415: softmax_tile
    int32_t t415 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t415, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t415, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t415);

    // Task 416: attention_output_tile
    int32_t t416 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t416, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t416, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t416, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t416);

    // Task 417: linear_tile
    int32_t t417 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t417, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t417, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t417, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t417);

    // Task 418: residual_add_tile
    int32_t t418 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t418, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t418, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t418, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t418);

    // Task 419: rmsnorm_tile
    int32_t t419 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t419, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t419, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t419, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t419);

    // Task 420: linear_tile
    int32_t t420 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t420, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t420, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t420, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t420);

    // Task 421: linear_tile
    int32_t t421 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t421, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t421, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t421, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t421);

    // Task 422: swiglu_tile
    int32_t t422 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t422, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t422, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t422, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t422);

    // Task 423: linear_tile
    int32_t t423 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t423, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t423, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t423, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t423);

    // Task 424: residual_add_tile
    int32_t t424 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t424, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t424, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t424, output, 0, 0, 8, 8);
    pto_task_submit(rt, t424);

    // ========== Tile 25 ==========
    // Task 425: rmsnorm_tile
    int32_t t425 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t425, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t425, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t425, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t425);

    // Task 426: linear_tile
    int32_t t426 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t426, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t426, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t426, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t426);

    // Task 427: linear_tile
    int32_t t427 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t427, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t427, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t427, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t427);

    // Task 428: linear_tile
    int32_t t428 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t428, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t428, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t428, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t428);

    // Task 429: rope_tile
    int32_t t429 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t429, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t429, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t429, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t429, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t429);

    // Task 430: rope_tile
    int32_t t430 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t430, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t430, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t430, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t430, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t430);

    // Task 431: attention_score_tile
    int32_t t431 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t431, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t431, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t431, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t431);

    // Task 432: softmax_tile
    int32_t t432 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t432, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t432, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t432);

    // Task 433: attention_output_tile
    int32_t t433 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t433, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t433, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t433, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t433);

    // Task 434: linear_tile
    int32_t t434 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t434, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t434, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t434, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t434);

    // Task 435: residual_add_tile
    int32_t t435 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t435, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t435, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t435, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t435);

    // Task 436: rmsnorm_tile
    int32_t t436 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t436, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t436, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t436, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t436);

    // Task 437: linear_tile
    int32_t t437 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t437, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t437, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t437, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t437);

    // Task 438: linear_tile
    int32_t t438 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t438, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t438, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t438, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t438);

    // Task 439: swiglu_tile
    int32_t t439 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t439, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t439, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t439, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t439);

    // Task 440: linear_tile
    int32_t t440 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t440, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t440, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t440, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t440);

    // Task 441: residual_add_tile
    int32_t t441 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t441, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t441, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t441, output, 0, 0, 8, 8);
    pto_task_submit(rt, t441);

    // ========== Tile 26 ==========
    // Task 442: rmsnorm_tile
    int32_t t442 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t442, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t442, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t442, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t442);

    // Task 443: linear_tile
    int32_t t443 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t443, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t443, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t443, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t443);

    // Task 444: linear_tile
    int32_t t444 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t444, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t444, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t444, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t444);

    // Task 445: linear_tile
    int32_t t445 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t445, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t445, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t445, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t445);

    // Task 446: rope_tile
    int32_t t446 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t446, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t446, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t446, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t446, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t446);

    // Task 447: rope_tile
    int32_t t447 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t447, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t447, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t447, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t447, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t447);

    // Task 448: attention_score_tile
    int32_t t448 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t448, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t448, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t448, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t448);

    // Task 449: softmax_tile
    int32_t t449 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t449, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t449, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t449);

    // Task 450: attention_output_tile
    int32_t t450 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t450, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t450, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t450, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t450);

    // Task 451: linear_tile
    int32_t t451 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t451, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t451, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t451, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t451);

    // Task 452: residual_add_tile
    int32_t t452 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t452, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t452, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t452, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t452);

    // Task 453: rmsnorm_tile
    int32_t t453 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t453, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t453, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t453, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t453);

    // Task 454: linear_tile
    int32_t t454 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t454, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t454, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t454, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t454);

    // Task 455: linear_tile
    int32_t t455 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t455, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t455, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t455, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t455);

    // Task 456: swiglu_tile
    int32_t t456 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t456, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t456, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t456, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t456);

    // Task 457: linear_tile
    int32_t t457 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t457, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t457, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t457, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t457);

    // Task 458: residual_add_tile
    int32_t t458 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t458, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t458, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t458, output, 0, 0, 8, 8);
    pto_task_submit(rt, t458);

    // ========== Tile 27 ==========
    // Task 459: rmsnorm_tile
    int32_t t459 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t459, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t459, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t459, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t459);

    // Task 460: linear_tile
    int32_t t460 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t460, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t460, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t460, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t460);

    // Task 461: linear_tile
    int32_t t461 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t461, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t461, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t461, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t461);

    // Task 462: linear_tile
    int32_t t462 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t462, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t462, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t462, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t462);

    // Task 463: rope_tile
    int32_t t463 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t463, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t463, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t463, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t463, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t463);

    // Task 464: rope_tile
    int32_t t464 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t464, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t464, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t464, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t464, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t464);

    // Task 465: attention_score_tile
    int32_t t465 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t465, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t465, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t465, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t465);

    // Task 466: softmax_tile
    int32_t t466 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t466, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t466, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t466);

    // Task 467: attention_output_tile
    int32_t t467 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t467, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t467, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t467, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t467);

    // Task 468: linear_tile
    int32_t t468 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t468, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t468, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t468, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t468);

    // Task 469: residual_add_tile
    int32_t t469 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t469, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t469, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t469, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t469);

    // Task 470: rmsnorm_tile
    int32_t t470 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t470, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t470, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t470, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t470);

    // Task 471: linear_tile
    int32_t t471 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t471, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t471, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t471, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t471);

    // Task 472: linear_tile
    int32_t t472 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t472, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t472, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t472, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t472);

    // Task 473: swiglu_tile
    int32_t t473 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t473, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t473, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t473, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t473);

    // Task 474: linear_tile
    int32_t t474 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t474, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t474, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t474, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t474);

    // Task 475: residual_add_tile
    int32_t t475 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t475, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t475, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t475, output, 0, 0, 8, 8);
    pto_task_submit(rt, t475);

    // ========== Tile 28 ==========
    // Task 476: rmsnorm_tile
    int32_t t476 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t476, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t476, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t476, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t476);

    // Task 477: linear_tile
    int32_t t477 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t477, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t477, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t477, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t477);

    // Task 478: linear_tile
    int32_t t478 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t478, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t478, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t478, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t478);

    // Task 479: linear_tile
    int32_t t479 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t479, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t479, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t479, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t479);

    // Task 480: rope_tile
    int32_t t480 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t480, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t480, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t480, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t480, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t480);

    // Task 481: rope_tile
    int32_t t481 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t481, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t481, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t481, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t481, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t481);

    // Task 482: attention_score_tile
    int32_t t482 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t482, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t482, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t482, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t482);

    // Task 483: softmax_tile
    int32_t t483 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t483, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t483, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t483);

    // Task 484: attention_output_tile
    int32_t t484 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t484, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t484, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t484, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t484);

    // Task 485: linear_tile
    int32_t t485 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t485, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t485, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t485, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t485);

    // Task 486: residual_add_tile
    int32_t t486 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t486, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t486, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t486, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t486);

    // Task 487: rmsnorm_tile
    int32_t t487 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t487, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t487, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t487, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t487);

    // Task 488: linear_tile
    int32_t t488 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t488, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t488, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t488, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t488);

    // Task 489: linear_tile
    int32_t t489 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t489, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t489, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t489, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t489);

    // Task 490: swiglu_tile
    int32_t t490 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t490, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t490, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t490, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t490);

    // Task 491: linear_tile
    int32_t t491 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t491, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t491, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t491, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t491);

    // Task 492: residual_add_tile
    int32_t t492 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t492, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t492, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t492, output, 0, 0, 8, 8);
    pto_task_submit(rt, t492);

    // ========== Tile 29 ==========
    // Task 493: rmsnorm_tile
    int32_t t493 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t493, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t493, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t493, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t493);

    // Task 494: linear_tile
    int32_t t494 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t494, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t494, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t494, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t494);

    // Task 495: linear_tile
    int32_t t495 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t495, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t495, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t495, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t495);

    // Task 496: linear_tile
    int32_t t496 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t496, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t496, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t496, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t496);

    // Task 497: rope_tile
    int32_t t497 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t497, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t497, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t497, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t497, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t497);

    // Task 498: rope_tile
    int32_t t498 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t498, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t498, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t498, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t498, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t498);

    // Task 499: attention_score_tile
    int32_t t499 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t499, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t499, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t499, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t499);

    // Task 500: softmax_tile
    int32_t t500 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t500, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t500, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t500);

    // Task 501: attention_output_tile
    int32_t t501 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t501, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t501, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t501, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t501);

    // Task 502: linear_tile
    int32_t t502 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t502, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t502, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t502, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t502);

    // Task 503: residual_add_tile
    int32_t t503 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t503, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t503, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t503, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t503);

    // Task 504: rmsnorm_tile
    int32_t t504 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t504, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t504, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t504, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t504);

    // Task 505: linear_tile
    int32_t t505 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t505, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t505, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t505, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t505);

    // Task 506: linear_tile
    int32_t t506 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t506, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t506, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t506, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t506);

    // Task 507: swiglu_tile
    int32_t t507 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t507, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t507, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t507, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t507);

    // Task 508: linear_tile
    int32_t t508 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t508, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t508, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t508, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t508);

    // Task 509: residual_add_tile
    int32_t t509 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t509, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t509, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t509, output, 0, 0, 8, 8);
    pto_task_submit(rt, t509);

    // ========== Tile 30 ==========
    // Task 510: rmsnorm_tile
    int32_t t510 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t510, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t510, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t510, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t510);

    // Task 511: linear_tile
    int32_t t511 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t511, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t511, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t511, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t511);

    // Task 512: linear_tile
    int32_t t512 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t512, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t512, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t512, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t512);

    // Task 513: linear_tile
    int32_t t513 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t513, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t513, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t513, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t513);

    // Task 514: rope_tile
    int32_t t514 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t514, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t514, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t514, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t514, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t514);

    // Task 515: rope_tile
    int32_t t515 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t515, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t515, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t515, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t515, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t515);

    // Task 516: attention_score_tile
    int32_t t516 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t516, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t516, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t516, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t516);

    // Task 517: softmax_tile
    int32_t t517 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t517, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t517, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t517);

    // Task 518: attention_output_tile
    int32_t t518 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t518, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t518, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t518, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t518);

    // Task 519: linear_tile
    int32_t t519 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t519, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t519, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t519, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t519);

    // Task 520: residual_add_tile
    int32_t t520 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t520, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t520, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t520, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t520);

    // Task 521: rmsnorm_tile
    int32_t t521 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t521, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t521, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t521, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t521);

    // Task 522: linear_tile
    int32_t t522 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t522, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t522, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t522, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t522);

    // Task 523: linear_tile
    int32_t t523 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t523, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t523, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t523, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t523);

    // Task 524: swiglu_tile
    int32_t t524 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t524, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t524, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t524, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t524);

    // Task 525: linear_tile
    int32_t t525 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t525, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t525, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t525, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t525);

    // Task 526: residual_add_tile
    int32_t t526 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t526, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t526, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t526, output, 0, 0, 8, 8);
    pto_task_submit(rt, t526);

    // ========== Tile 31 ==========
    // Task 527: rmsnorm_tile
    int32_t t527 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t527, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t527, attn_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t527, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t527);

    // Task 528: linear_tile
    int32_t t528 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t528, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t528, wq, 0, 0, 8, 8);
    pto_task_add_output(rt, t528, temp_q, 0, 0, 8, 8);
    pto_task_submit(rt, t528);

    // Task 529: linear_tile
    int32_t t529 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t529, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t529, wk, 0, 0, 8, 8);
    pto_task_add_output(rt, t529, temp_k, 0, 0, 8, 8);
    pto_task_submit(rt, t529);

    // Task 530: linear_tile
    int32_t t530 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t530, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t530, wv, 0, 0, 8, 8);
    pto_task_add_output(rt, t530, temp_v, 0, 0, 8, 8);
    pto_task_submit(rt, t530);

    // Task 531: rope_tile
    int32_t t531 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t531, temp_q, 0, 0, 8, 8);
    pto_task_add_input(rt, t531, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t531, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t531, temp_q_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t531);

    // Task 532: rope_tile
    int32_t t532 = pto_task_alloc(rt, "rope_tile", NULL);
    pto_task_add_input(rt, t532, temp_k, 0, 0, 8, 8);
    pto_task_add_input(rt, t532, cos_cache, 0, 0, 8, 8);
    pto_task_add_input(rt, t532, sin_cache, 0, 0, 8, 8);
    pto_task_add_output(rt, t532, temp_k_rope, 0, 0, 8, 8);
    pto_task_submit(rt, t532);

    // Task 533: attention_score_tile
    int32_t t533 = pto_task_alloc(rt, "attention_score_tile", NULL);
    pto_task_add_input(rt, t533, temp_q_rope, 0, 0, 8, 8);
    pto_task_add_input(rt, t533, temp_k_rope, 0, 0, 8, 8);
    pto_task_add_output(rt, t533, temp_scores, 0, 0, 8, 8);
    pto_task_submit(rt, t533);

    // Task 534: softmax_tile
    int32_t t534 = pto_task_alloc(rt, "softmax_tile", NULL);
    pto_task_add_input(rt, t534, temp_scores, 0, 0, 8, 8);
    pto_task_add_output(rt, t534, temp_attn_weights, 0, 0, 8, 8);
    pto_task_submit(rt, t534);

    // Task 535: attention_output_tile
    int32_t t535 = pto_task_alloc(rt, "attention_output_tile", NULL);
    pto_task_add_input(rt, t535, temp_attn_weights, 0, 0, 8, 8);
    pto_task_add_input(rt, t535, temp_v, 0, 0, 8, 8);
    pto_task_add_output(rt, t535, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t535);

    // Task 536: linear_tile
    int32_t t536 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t536, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t536, wo, 0, 0, 8, 8);
    pto_task_add_output(rt, t536, temp_attn_out, 0, 0, 8, 8);
    pto_task_submit(rt, t536);

    // Task 537: residual_add_tile
    int32_t t537 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t537, temp_attn_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t537, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t537, temp_residual1, 0, 0, 8, 8);
    pto_task_submit(rt, t537);

    // Task 538: rmsnorm_tile
    int32_t t538 = pto_task_alloc(rt, "rmsnorm_tile", NULL);
    pto_task_add_input(rt, t538, temp_residual1, 0, 0, 8, 8);
    pto_task_add_input(rt, t538, mlp_norm_weights, 0, 0, 8, 8);
    pto_task_add_output(rt, t538, temp_norm, 0, 0, 8, 8);
    pto_task_submit(rt, t538);

    // Task 539: linear_tile
    int32_t t539 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t539, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t539, w_gate, 0, 0, 8, 8);
    pto_task_add_output(rt, t539, temp_gate, 0, 0, 8, 8);
    pto_task_submit(rt, t539);

    // Task 540: linear_tile
    int32_t t540 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t540, temp_norm, 0, 0, 8, 8);
    pto_task_add_input(rt, t540, w_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t540, temp_up, 0, 0, 8, 8);
    pto_task_submit(rt, t540);

    // Task 541: swiglu_tile
    int32_t t541 = pto_task_alloc(rt, "swiglu_tile", NULL);
    pto_task_add_input(rt, t541, temp_gate, 0, 0, 8, 8);
    pto_task_add_input(rt, t541, temp_up, 0, 0, 8, 8);
    pto_task_add_output(rt, t541, temp_swiglu, 0, 0, 8, 8);
    pto_task_submit(rt, t541);

    // Task 542: linear_tile
    int32_t t542 = pto_task_alloc(rt, "linear_tile", NULL);
    pto_task_add_input(rt, t542, temp_swiglu, 0, 0, 8, 8);
    pto_task_add_input(rt, t542, w_down, 0, 0, 8, 8);
    pto_task_add_output(rt, t542, temp_mlp_out, 0, 0, 8, 8);
    pto_task_submit(rt, t542);

    // Task 543: residual_add_tile
    int32_t t543 = pto_task_alloc(rt, "residual_add_tile", NULL);
    pto_task_add_input(rt, t543, temp_mlp_out, 0, 0, 8, 8);
    pto_task_add_input(rt, t543, temp_residual1, 0, 0, 8, 8);
    pto_task_add_output(rt, t543, output, 0, 0, 8, 8);
    pto_task_submit(rt, t543);

}

int main(int argc, char** argv) {
    PTORuntime rt;
    pto_runtime_init(&rt);

    build_task_graph(&rt);

    printf("\n");
    pto_runtime_dump_stdout(&rt);
    pto_runtime_dump(&rt, "llama_layer_dynamic_seq256_task_graph.txt");

    pto_runtime_shutdown(&rt);
    return 0;
}