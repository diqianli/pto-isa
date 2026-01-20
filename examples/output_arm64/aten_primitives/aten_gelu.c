// PTO Program: aten_gelu
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_gelu(float* input, float* output) {
    float x[1][4096];
    float scaled_x[1][4096];
    float neg_scaled[1][4096];
    float exp_neg[1][4096];
    float one_plus[1][4096];
    float sigmoid_out[1][4096];
    float result[1][4096];

    // Loop fusion: 15 loop overheads saved

    // FUSED LOOP (16 ops): x=TLOAD(input,0,0); scaled_x=TMULS(x,1.702f); neg_scaled=TNEG(scaled_x); exp_neg=TEXP(neg_scaled); one_plus=TADDS(exp_neg,1.0f); sigmoid_out=TRECIP(one_plus); result=TMUL(x,sigmoid_out); output=TSTORE(result,0,0); x=TLOAD(input,0,0); scaled_x=TMULS(x,1.702f); neg_scaled=TNEG(scaled_x); exp_neg=TEXP(neg_scaled); one_plus=TADDS(exp_neg,1.0f); sigmoid_out=TRECIP(one_plus); result=TMUL(x,sigmoid_out); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.702f);
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl2 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vmulq_f32(_v3, _vs0);
            vst1q_f32(&scaled_x[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&scaled_x[_row][_col]);
            float32x4_t _vr6 = vnegq_f32(_v5);
            vst1q_f32(&neg_scaled[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&neg_scaled[_row][_col]);
            float32x4_t _vr8 = _v7;
            vst1q_f32(&exp_neg[_row][_col], _vr8);
            float32x4_t _v9 = vld1q_f32(&exp_neg[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v9, _vs1);
            vst1q_f32(&one_plus[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr12 = _v11;
            vst1q_f32(&sigmoid_out[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v14 = vld1q_f32(&sigmoid_out[_row][_col]);
            float32x4_t _vr15 = vmulq_f32(_v13, _v14);
            vst1q_f32(&result[_row][_col], _vr15);
            float32x4_t _vs16 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs16);
            float32x4_t _vl17 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl17);
            float32x4_t _v18 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr19 = vmulq_f32(_v18, _vs0);
            vst1q_f32(&scaled_x[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&scaled_x[_row][_col]);
            float32x4_t _vr21 = vnegq_f32(_v20);
            vst1q_f32(&neg_scaled[_row][_col], _vr21);
            float32x4_t _v22 = vld1q_f32(&neg_scaled[_row][_col]);
            float32x4_t _vr23 = _v22;
            vst1q_f32(&exp_neg[_row][_col], _vr23);
            float32x4_t _v24 = vld1q_f32(&exp_neg[_row][_col]);
            float32x4_t _vr25 = vaddq_f32(_v24, _vs1);
            vst1q_f32(&one_plus[_row][_col], _vr25);
            float32x4_t _v26 = vld1q_f32(&one_plus[_row][_col]);
            float32x4_t _vr27 = _v26;
            vst1q_f32(&sigmoid_out[_row][_col], _vr27);
            float32x4_t _v28 = vld1q_f32(&x[_row][_col]);
            float32x4_t _v29 = vld1q_f32(&sigmoid_out[_row][_col]);
            float32x4_t _vr30 = vmulq_f32(_v28, _v29);
            vst1q_f32(&result[_row][_col], _vr30);
            float32x4_t _vs31 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs31);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input[_row * 4096 + _col];
            scaled_x[_row][_col] = x[_row][_col] * 1.702f;
            neg_scaled[_row][_col] = -scaled_x[_row][_col];
            exp_neg[_row][_col] = expf(neg_scaled[_row][_col]);
            one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
            sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
            x[_row][_col] = input[_row * 4096 + _col];
            scaled_x[_row][_col] = x[_row][_col] * 1.702f;
            neg_scaled[_row][_col] = -scaled_x[_row][_col];
            exp_neg[_row][_col] = expf(neg_scaled[_row][_col]);
            one_plus[_row][_col] = exp_neg[_row][_col] + 1.0f;
            sigmoid_out[_row][_col] = 1.0f / one_plus[_row][_col];
            result[_row][_col] = x[_row][_col] * sigmoid_out[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
        }
    }

}