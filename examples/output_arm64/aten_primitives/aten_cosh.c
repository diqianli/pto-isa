// PTO Program: aten_cosh
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_cosh(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    float x[1][4096];
    float neg_x[1][4096];
    float exp_x[1][4096];
    float exp_neg_x[1][4096];
    float sum[1][4096];
    float result[1][4096];

    // Loop fusion: 12 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (7 ops): x=TLOAD(input,tile_idx,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); sum=TADD(exp_x,exp_neg_x); result=TDIVS(sum,2.0f); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs0 = vdupq_n_f32(2.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl1 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
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
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs13);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
                exp_x[_row][_col] = expf(x[_row][_col]);
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
                sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
                result[_row][_col] = sum[_row][_col] / 2.0f;
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (7 ops): x=TLOAD(input,num_full_tiles,0); exp_x=TEXP(x); neg_x=TNEG(x); exp_neg_x=TEXP(neg_x); sum=TADD(exp_x,exp_neg_x); result=TDIVS(sum,2.0f); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs14 = vdupq_n_f32(2.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl15 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl15);
                float32x4_t _v16 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr17 = _v16;
                vst1q_f32(&exp_x[_row][_col], _vr17);
                float32x4_t _v18 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr19 = vnegq_f32(_v18);
                vst1q_f32(&neg_x[_row][_col], _vr19);
                float32x4_t _v20 = vld1q_f32(&neg_x[_row][_col]);
                float32x4_t _vr21 = _v20;
                vst1q_f32(&exp_neg_x[_row][_col], _vr21);
                float32x4_t _v22 = vld1q_f32(&exp_x[_row][_col]);
                float32x4_t _v23 = vld1q_f32(&exp_neg_x[_row][_col]);
                float32x4_t _vr24 = vaddq_f32(_v22, _v23);
                vst1q_f32(&sum[_row][_col], _vr24);
                float32x4_t _v25 = vld1q_f32(&sum[_row][_col]);
                float32x4_t _vr26 = vdivq_f32(_v25, _vs14);
                vst1q_f32(&result[_row][_col], _vr26);
                float32x4_t _vs27 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs27);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
                exp_x[_row][_col] = expf(x[_row][_col]);
                neg_x[_row][_col] = -x[_row][_col];
                exp_neg_x[_row][_col] = expf(neg_x[_row][_col]);
                sum[_row][_col] = exp_x[_row][_col] + exp_neg_x[_row][_col];
                result[_row][_col] = sum[_row][_col] / 2.0f;
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}