// PTO Program: aten_tanh
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_tanh(float* input, float* output) {
    float x[1][4096];
    float exp_x[1][4096];
    float exp_neg_x[1][4096];
    float neg_x[1][4096];
    float numerator[1][4096];
    float denominator[1][4096];
    float result[1][4096];

    // Loop fusion: 15 loop overheads saved

    // FUSED LOOP (16 ops): x=TLOAD(input,0,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); numerator=TSUB(exp_x,exp_neg_x); denominator=TADD(exp_x,exp_neg_x); result=TDIV(numerator,denominator); output=TSTORE(result,0,0); x=TLOAD(input,0,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); numerator=TSUB(exp_x,exp_neg_x); denominator=TADD(exp_x,exp_neg_x); result=TDIV(numerator,denominator); output=TSTORE(result,0,0)
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl0 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl0);
            float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr2 = _v1;
            vst1q_f32(&exp_x[_row][_col], _vr2);
            float32x4_t _v3 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr4 = vnegq_f32(_v3);
            vst1q_f32(&neg_x[_row][_col], _vr4);
            float32x4_t _v5 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr6 = _v5;
            vst1q_f32(&exp_neg_x[_row][_col], _vr6);
            float32x4_t _v7 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v8 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr9 = vsubq_f32(_v7, _v8);
            vst1q_f32(&numerator[_row][_col], _vr9);
            float32x4_t _v10 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v11 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr12 = vaddq_f32(_v10, _v11);
            vst1q_f32(&denominator[_row][_col], _vr12);
            float32x4_t _v13 = vld1q_f32(&numerator[_row][_col]);
            float32x4_t _v14 = vld1q_f32(&denominator[_row][_col]);
            float32x4_t _vr15 = vdivq_f32(_v13, _v14);
            vst1q_f32(&result[_row][_col], _vr15);
            float32x4_t _vs16 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs16);
            float32x4_t _vl17 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl17);
            float32x4_t _v18 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr19 = _v18;
            vst1q_f32(&exp_x[_row][_col], _vr19);
            float32x4_t _v20 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr21 = vnegq_f32(_v20);
            vst1q_f32(&neg_x[_row][_col], _vr21);
            float32x4_t _v22 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr23 = _v22;
            vst1q_f32(&exp_neg_x[_row][_col], _vr23);
            float32x4_t _v24 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v25 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr26 = vsubq_f32(_v24, _v25);
            vst1q_f32(&numerator[_row][_col], _vr26);
            float32x4_t _v27 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v28 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr29 = vaddq_f32(_v27, _v28);
            vst1q_f32(&denominator[_row][_col], _vr29);
            float32x4_t _v30 = vld1q_f32(&numerator[_row][_col]);
            float32x4_t _v31 = vld1q_f32(&denominator[_row][_col]);
            float32x4_t _vr32 = vdivq_f32(_v30, _v31);
            vst1q_f32(&result[_row][_col], _vr32);
            float32x4_t _vs33 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs33);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input[_row * 4096 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            numerator[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
            denominator[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
            result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
            x[_row][_col] = input[_row * 4096 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            numerator[_row][_col] = exp_x[_row][_col] - exp_neg_x[_row][_col];
            denominator[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
            result[_row][_col] = numerator[_row][_col] / denominator[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
        }
    }

}