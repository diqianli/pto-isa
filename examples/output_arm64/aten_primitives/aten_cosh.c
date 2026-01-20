// PTO Program: aten_cosh
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_cosh(float* input, float* output) {
    float x[1][4096];
    float neg_x[1][4096];
    float exp_x[1][4096];
    float exp_neg_x[1][4096];
    float sum[1][4096];
    float result[1][4096];

    // Loop fusion: 13 loop overheads saved

    // FUSED LOOP (14 ops): x=TLOAD(input,0,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); sum=TADD(exp_x,exp_neg_x); result=TDIVS(sum,2.0f); output=TSTORE(result,0,0); x=TLOAD(input,0,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); sum=TADD(exp_x,exp_neg_x); result=TDIVS(sum,2.0f); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(2.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = _v2;
            vst1q_f32(&exp_x[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr5 = vnegq_f32(_v4);
            vst1q_f32(&neg_x[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr7 = _v6;
            vst1q_f32(&exp_neg_x[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v9 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr10 = vaddq_f32(_v8, _v9);
            vst1q_f32(&sum[_row][_col], _vr10);
            float32x4_t _v11 = vld1q_f32(&sum[_row][_col]);
            float32x4_t _vr12 = vdivq_f32(_v11, _vs0);
            vst1q_f32(&result[_row][_col], _vr12);
            float32x4_t _vs13 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs13);
            float32x4_t _vl14 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl14);
            float32x4_t _v15 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr16 = _v15;
            vst1q_f32(&exp_x[_row][_col], _vr16);
            float32x4_t _v17 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr18 = vnegq_f32(_v17);
            vst1q_f32(&neg_x[_row][_col], _vr18);
            float32x4_t _v19 = vld1q_f32(&neg_x[_row][_col]);
            float32x4_t _vr20 = _v19;
            vst1q_f32(&exp_neg_x[_row][_col], _vr20);
            float32x4_t _v21 = vld1q_f32(&exp_x[_row][_col]);
            float32x4_t _v22 = vld1q_f32(&exp_neg_x[_row][_col]);
            float32x4_t _vr23 = vaddq_f32(_v21, _v22);
            vst1q_f32(&sum[_row][_col], _vr23);
            float32x4_t _v24 = vld1q_f32(&sum[_row][_col]);
            float32x4_t _vr25 = vdivq_f32(_v24, _vs0);
            vst1q_f32(&result[_row][_col], _vr25);
            float32x4_t _vs26 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs26);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input[_row * 4096 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
            result[_row][_col] = sum[_row][_col] / 2.0f;
            output[_row * 4096 + _col] = result[_row][_col];
            x[_row][_col] = input[_row * 4096 + _col];
            exp_x[_row][_col] = expf(x[_row][_col]);
            neg_x[_row][_col] = -x[_row][_col];
            exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
            sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
            result[_row][_col] = sum[_row][_col] / 2.0f;
            output[_row * 4096 + _col] = result[_row][_col];
        }
    }

}