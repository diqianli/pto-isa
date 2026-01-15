// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

float x[8][8];
float x_squared[8][8];
float term[8][8];
float result[8][8];

// Loop fusion: 20 loop overheads saved

// TLOAD: x = tload input[0, 0]
for (int _row = 0; _row < 8; _row++) {
    for (int _col = 0; _col < 8; _col++) {
        x[_row][_col] = input[(_row + 0) * 8 + (_col + 0)];
    }
}

// FUSED LOOP (21 ops): result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,110.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,156.0f); result=TADD(result,term)
float32x4_t _vs0 = vdupq_n_f32(1.0f);
float32x4_t _vs1 = vdupq_n_f32(6.0f);
float32x4_t _vs2 = vdupq_n_f32(20.0f);
float32x4_t _vs3 = vdupq_n_f32(42.0f);
float32x4_t _vs4 = vdupq_n_f32(72.0f);
float32x4_t _vs5 = vdupq_n_f32(110.0f);
float32x4_t _vs6 = vdupq_n_f32(156.0f);
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v7 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr8 = vmulq_f32(_v7, _vs0);
        vst1q_f32(&result[_row][_col], _vr8);
        float32x4_t _v9 = vld1q_f32(&x[_row][_col]);
        float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr11 = vmulq_f32(_v9, _v10);
        vst1q_f32(&x_squared[_row][_col], _vr11);
        float32x4_t _v12 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr13 = vmulq_f32(_v12, _vs0);
        vst1q_f32(&term[_row][_col], _vr13);
        float32x4_t _v14 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v15 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr16 = vmulq_f32(_v14, _v15);
        vst1q_f32(&term[_row][_col], _vr16);
        float32x4_t _v17 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr18 = vdivq_f32(_v17, _vs1);
        vst1q_f32(&term[_row][_col], _vr18);
        float32x4_t _v19 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v20 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr21 = vaddq_f32(_v19, _v20);
        vst1q_f32(&result[_row][_col], _vr21);
        float32x4_t _v22 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v23 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr24 = vmulq_f32(_v22, _v23);
        vst1q_f32(&term[_row][_col], _vr24);
        float32x4_t _v25 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr26 = vdivq_f32(_v25, _vs2);
        vst1q_f32(&term[_row][_col], _vr26);
        float32x4_t _v27 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v28 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr29 = vaddq_f32(_v27, _v28);
        vst1q_f32(&result[_row][_col], _vr29);
        float32x4_t _v30 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v31 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr32 = vmulq_f32(_v30, _v31);
        vst1q_f32(&term[_row][_col], _vr32);
        float32x4_t _v33 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr34 = vdivq_f32(_v33, _vs3);
        vst1q_f32(&term[_row][_col], _vr34);
        float32x4_t _v35 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v36 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr37 = vaddq_f32(_v35, _v36);
        vst1q_f32(&result[_row][_col], _vr37);
        float32x4_t _v38 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v39 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr40 = vmulq_f32(_v38, _v39);
        vst1q_f32(&term[_row][_col], _vr40);
        float32x4_t _v41 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr42 = vdivq_f32(_v41, _vs4);
        vst1q_f32(&term[_row][_col], _vr42);
        float32x4_t _v43 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v44 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr45 = vaddq_f32(_v43, _v44);
        vst1q_f32(&result[_row][_col], _vr45);
        float32x4_t _v46 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v47 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr48 = vmulq_f32(_v46, _v47);
        vst1q_f32(&term[_row][_col], _vr48);
        float32x4_t _v49 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr50 = vdivq_f32(_v49, _vs5);
        vst1q_f32(&term[_row][_col], _vr50);
        float32x4_t _v51 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v52 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr53 = vaddq_f32(_v51, _v52);
        vst1q_f32(&result[_row][_col], _vr53);
        float32x4_t _v54 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v55 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr56 = vmulq_f32(_v54, _v55);
        vst1q_f32(&term[_row][_col], _vr56);
        float32x4_t _v57 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr58 = vdivq_f32(_v57, _vs6);
        vst1q_f32(&term[_row][_col], _vr58);
        float32x4_t _v59 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v60 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr61 = vaddq_f32(_v59, _v60);
        vst1q_f32(&result[_row][_col], _vr61);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = x[_row][_col] * 1.0f;
        x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
        term[_row][_col] = x[_row][_col] * 1.0f;
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 6.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 20.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 42.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 72.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 110.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
        term[_row][_col] = term[_row][_col] / 156.0f;
        result[_row][_col] = result[_row][_col] + term[_row][_col];
    }
}

// TSTORE: tstore result, output[0, 0]
for (int _row = 0; _row < 8; _row++) {
    for (int _col = 0; _col < 8; _col++) {
        output[(_row + 0) * 8 + (_col + 0)] = result[_row][_col];
    }
}
