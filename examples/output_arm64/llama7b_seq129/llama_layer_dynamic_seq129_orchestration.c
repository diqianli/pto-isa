/**
 * LLaMA 7B Layer Orchestration - seq_len=129
 * num_full_tiles: 16
 * tail_rows: 1
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

    // ========== Tail (1 rows) ==========
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

}

int main(int argc, char** argv) {
    PTORuntime rt;
    pto_runtime_init(&rt);

    build_task_graph(&rt);

    printf("\n");
    pto_runtime_dump_stdout(&rt);
    pto_runtime_dump(&rt, "llama_layer_dynamic_seq129_task_graph.txt");

    pto_runtime_shutdown(&rt);
    return 0;
}