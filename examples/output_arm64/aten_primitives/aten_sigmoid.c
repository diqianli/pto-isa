// PTO Program: aten_sigmoid
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_sigmoid(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    float x[1][4096];
    float t1[1][4096];
    float t2[1][4096];
    float t3[1][4096];
    float result[1][4096];

    // Loop fusion: 10 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (6 ops): x=TLOAD(input,tile_idx,0); t1=TNEG(x); t2=TEXP(t1); t3=TADDS(t2,1.0f); result=TRECIP(t3); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs0 = vdupq_n_f32(1.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl1 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
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
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs10);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
                t1[_row][_col] = -x[_row][_col];
                t2[_row][_col] = expf(t1[_row][_col]);
                t3[_row][_col] = t2[_row][_col] + 1.0f;
                result[_row][_col] = 1.0f / t3[_row][_col];
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (6 ops): x=TLOAD(input,num_full_tiles,0); t1=TNEG(x); t2=TEXP(t1); t3=TADDS(t2,1.0f); result=TRECIP(t3); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs11 = vdupq_n_f32(1.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl12 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl12);
                float32x4_t _v13 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr14 = vnegq_f32(_v13);
                vst1q_f32(&t1[_row][_col], _vr14);
                float32x4_t _v15 = vld1q_f32(&t1[_row][_col]);
                float32x4_t _vr16 = _v15;
                vst1q_f32(&t2[_row][_col], _vr16);
                float32x4_t _v17 = vld1q_f32(&t2[_row][_col]);
                float32x4_t _vr18 = vaddq_f32(_v17, _vs11);
                vst1q_f32(&t3[_row][_col], _vr18);
                float32x4_t _v19 = vld1q_f32(&t3[_row][_col]);
                float32x4_t _vr20 = _v19;
                vst1q_f32(&result[_row][_col], _vr20);
                float32x4_t _vs21 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs21);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
                t1[_row][_col] = -x[_row][_col];
                t2[_row][_col] = expf(t1[_row][_col]);
                t3[_row][_col] = t2[_row][_col] + 1.0f;
                result[_row][_col] = 1.0f / t3[_row][_col];
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}