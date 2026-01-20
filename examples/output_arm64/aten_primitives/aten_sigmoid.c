// PTO Program: aten_sigmoid
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_sigmoid(float* input, float* output) {
    float x[1][4096];
    float t1[1][4096];
    float t2[1][4096];
    float t3[1][4096];
    float result[1][4096];

    // Loop fusion: 11 loop overheads saved

    // FUSED LOOP (12 ops): x=TLOAD(input,0,0); t1=TNEG(x); t2=TEXP(t1); t3=TADDS(t2,1.0f); result=TRECIP(t3); output=TSTORE(result,0,0); x=TLOAD(input,0,0); t1=TNEG(x); t2=TEXP(t1); t3=TADDS(t2,1.0f); result=TRECIP(t3); output=TSTORE(result,0,0)
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 1; _row++) {
        int _col;
        // Vectorized loop
        for (_col = 0; _col + 4 <= 4096; _col += 4) {
            float32x4_t _vl1 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl1);
            float32x4_t _v2 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr3 = vnegq_f32(_v2);
            vst1q_f32(&t1[_row][_col], _vr3);
            float32x4_t _v4 = vld1q_f32(&t1[_row][_col]);
            float32x4_t _vr5 = _v4;
            vst1q_f32(&t2[_row][_col], _vr5);
            float32x4_t _v6 = vld1q_f32(&t2[_row][_col]);
            float32x4_t _vr7 = vaddq_f32(_v6, _vs0);
            vst1q_f32(&t3[_row][_col], _vr7);
            float32x4_t _v8 = vld1q_f32(&t3[_row][_col]);
            float32x4_t _vr9 = _v8;
            vst1q_f32(&result[_row][_col], _vr9);
            float32x4_t _vs10 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs10);
            float32x4_t _vl11 = vld1q_f32(&input[_row * 4096 + _col]);
            vst1q_f32(&x[_row][_col], _vl11);
            float32x4_t _v12 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr13 = vnegq_f32(_v12);
            vst1q_f32(&t1[_row][_col], _vr13);
            float32x4_t _v14 = vld1q_f32(&t1[_row][_col]);
            float32x4_t _vr15 = _v14;
            vst1q_f32(&t2[_row][_col], _vr15);
            float32x4_t _v16 = vld1q_f32(&t2[_row][_col]);
            float32x4_t _vr17 = vaddq_f32(_v16, _vs0);
            vst1q_f32(&t3[_row][_col], _vr17);
            float32x4_t _v18 = vld1q_f32(&t3[_row][_col]);
            float32x4_t _vr19 = _v18;
            vst1q_f32(&result[_row][_col], _vr19);
            float32x4_t _vs20 = vld1q_f32(&result[_row][_col]);
            vst1q_f32(&output[_row * 4096 + _col], _vs20);
        }
        // Scalar cleanup
        for (; _col < 4096; _col++) {
            x[_row][_col] = input[_row * 4096 + _col];
            t1[_row][_col] = -x[_row][_col];
            t2[_row][_col] = expf(t1[_row][_col]);
            t3[_row][_col] = t2[_row][_col] + 1.0f;
            result[_row][_col] = 1.0f / t3[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
            x[_row][_col] = input[_row * 4096 + _col];
            t1[_row][_col] = -x[_row][_col];
            t2[_row][_col] = expf(t1[_row][_col]);
            t3[_row][_col] = t2[_row][_col] + 1.0f;
            result[_row][_col] = 1.0f / t3[_row][_col];
            output[_row * 4096 + _col] = result[_row][_col];
        }
    }

}