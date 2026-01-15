// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

float x[8][8];
float x_squared[8][8];
float term[8][8];
float result[8][8];

// TLOAD: x = tload input[0, 0]
for (int _row = 0; _row < 8; _row++) {
    for (int _col = 0; _col < 8; _col++) {
        x[_row][_col] = input[(_row + 0) * 8 + (_col + 0)];
    }
}

// TMULS: result = tmuls x, 1.0f
{
    float32x4_t _vs0 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr = vmulq_f32(_v0, _vs0);
            vst1q_f32(&result[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            result[_row][_col] = x[_row][_col] * 1.0f;
        }
    }
}

// TMUL: x_squared = tmul x, x
for (int _row = 0; _row < 8; _row++) {
    int _col;
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&x[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr = vmulq_f32(_v0, _v1);
        vst1q_f32(&x_squared[_row][_col], _vr);
    }
    for (; _col < 8; _col++) {
        x_squared[_row][_col] = x[_row][_col] * x[_row][_col];
    }
}

// TMULS: term = tmuls x, 1.0f
{
    float32x4_t _vs1 = vdupq_n_f32(1.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&x[_row][_col]);
            float32x4_t _vr = vmulq_f32(_v0, _vs1);
            vst1q_f32(&term[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            term[_row][_col] = x[_row][_col] * 1.0f;
        }
    }
}

// TMUL: term = tmul term, x_squared
for (int _row = 0; _row < 8; _row++) {
    int _col;
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr = vmulq_f32(_v0, _v1);
        vst1q_f32(&term[_row][_col], _vr);
    }
    for (; _col < 8; _col++) {
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
    }
}

// TDIVS: term = tdivs term, 6.0f
{
    float32x4_t _vs2 = vdupq_n_f32(6.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr = vdivq_f32(_v0, _vs2);
            vst1q_f32(&term[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            term[_row][_col] = term[_row][_col] / 6.0f;
        }
    }
}

// TADD: result = tadd result, term
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr = vaddq_f32(_v0, _v1);
        vst1q_f32(&result[_row][_col], _vr);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = result[_row][_col] + term[_row][_col];
    }
}

// TMUL: term = tmul term, x_squared
for (int _row = 0; _row < 8; _row++) {
    int _col;
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr = vmulq_f32(_v0, _v1);
        vst1q_f32(&term[_row][_col], _vr);
    }
    for (; _col < 8; _col++) {
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
    }
}

// TDIVS: term = tdivs term, 20.0f
{
    float32x4_t _vs3 = vdupq_n_f32(20.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr = vdivq_f32(_v0, _vs3);
            vst1q_f32(&term[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            term[_row][_col] = term[_row][_col] / 20.0f;
        }
    }
}

// TADD: result = tadd result, term
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr = vaddq_f32(_v0, _v1);
        vst1q_f32(&result[_row][_col], _vr);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = result[_row][_col] + term[_row][_col];
    }
}

// TMUL: term = tmul term, x_squared
for (int _row = 0; _row < 8; _row++) {
    int _col;
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr = vmulq_f32(_v0, _v1);
        vst1q_f32(&term[_row][_col], _vr);
    }
    for (; _col < 8; _col++) {
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
    }
}

// TDIVS: term = tdivs term, 42.0f
{
    float32x4_t _vs4 = vdupq_n_f32(42.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr = vdivq_f32(_v0, _vs4);
            vst1q_f32(&term[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            term[_row][_col] = term[_row][_col] / 42.0f;
        }
    }
}

// TADD: result = tadd result, term
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr = vaddq_f32(_v0, _v1);
        vst1q_f32(&result[_row][_col], _vr);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = result[_row][_col] + term[_row][_col];
    }
}

// TMUL: term = tmul term, x_squared
for (int _row = 0; _row < 8; _row++) {
    int _col;
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr = vmulq_f32(_v0, _v1);
        vst1q_f32(&term[_row][_col], _vr);
    }
    for (; _col < 8; _col++) {
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
    }
}

// TDIVS: term = tdivs term, 72.0f
{
    float32x4_t _vs5 = vdupq_n_f32(72.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr = vdivq_f32(_v0, _vs5);
            vst1q_f32(&term[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            term[_row][_col] = term[_row][_col] / 72.0f;
        }
    }
}

// TADD: result = tadd result, term
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr = vaddq_f32(_v0, _v1);
        vst1q_f32(&result[_row][_col], _vr);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = result[_row][_col] + term[_row][_col];
    }
}

// TMUL: term = tmul term, x_squared
for (int _row = 0; _row < 8; _row++) {
    int _col;
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr = vmulq_f32(_v0, _v1);
        vst1q_f32(&term[_row][_col], _vr);
    }
    for (; _col < 8; _col++) {
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
    }
}

// TDIVS: term = tdivs term, 110.0f
{
    float32x4_t _vs6 = vdupq_n_f32(110.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr = vdivq_f32(_v0, _vs6);
            vst1q_f32(&term[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            term[_row][_col] = term[_row][_col] / 110.0f;
        }
    }
}

// TADD: result = tadd result, term
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr = vaddq_f32(_v0, _v1);
        vst1q_f32(&result[_row][_col], _vr);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = result[_row][_col] + term[_row][_col];
    }
}

// TMUL: term = tmul term, x_squared
for (int _row = 0; _row < 8; _row++) {
    int _col;
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr = vmulq_f32(_v0, _v1);
        vst1q_f32(&term[_row][_col], _vr);
    }
    for (; _col < 8; _col++) {
        term[_row][_col] = term[_row][_col] * x_squared[_row][_col];
    }
}

// TDIVS: term = tdivs term, 156.0f
{
    float32x4_t _vs7 = vdupq_n_f32(156.0f);
    for (int _row = 0; _row < 8; _row++) {
        int _col;
        for (_col = 0; _col + 4 <= 8; _col += 4) {
            float32x4_t _v0 = vld1q_f32(&term[_row][_col]);
            float32x4_t _vr = vdivq_f32(_v0, _vs7);
            vst1q_f32(&term[_row][_col], _vr);
        }
        for (; _col < 8; _col++) {
            term[_row][_col] = term[_row][_col] / 156.0f;
        }
    }
}

// TADD: result = tadd result, term
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr = vaddq_f32(_v0, _v1);
        vst1q_f32(&result[_row][_col], _vr);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = result[_row][_col] + term[_row][_col];
    }
}

// TSTORE: tstore result, output[0, 0]
for (int _row = 0; _row < 8; _row++) {
    for (int _col = 0; _col < 8; _col++) {
        output[(_row + 0) * 8 + (_col + 0)] = result[_row][_col];
    }
}
