// PTO Program: aten_sinh
// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

void aten_sinh(float* input, float* output, int32_t num_full_tiles, int32_t tail_elements, int32_t zero, int32_t tile_size) {
    float x[1][4096];
    float x_squared[1][4096];
    float term[1][4096];
    float result[1][4096];

    // Loop fusion: 32 loop overheads saved

    int tile_size = 4096;

    int zero = 0;

    for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

        // FUSED LOOP (17 ops): x=TLOAD(input,tile_idx,0); result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); output=TSTORE(result,tile_idx,0)
        float32x4_t _vs0 = vdupq_n_f32(1.0f);
        float32x4_t _vs1 = vdupq_n_f32(6.0f);
        float32x4_t _vs2 = vdupq_n_f32(20.0f);
        float32x4_t _vs3 = vdupq_n_f32(42.0f);
        float32x4_t _vs4 = vdupq_n_f32(72.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl5 = vld1q_f32(&input[(tile_idx) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl5);
                float32x4_t _v6 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr7 = vmulq_f32(_v6, _vs0);
                vst1q_f32(&result[_row][_col], _vr7);
                float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v9 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr10 = vmulq_f32(_v8, _v9);
                vst1q_f32(&x_squared[_row][_col], _vr10);
                float32x4_t _v11 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr12 = vmulq_f32(_v11, _vs0);
                vst1q_f32(&term[_row][_col], _vr12);
                float32x4_t _v13 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v14 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr15 = vmulq_f32(_v13, _v14);
                vst1q_f32(&term[_row][_col], _vr15);
                float32x4_t _v16 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr17 = vdivq_f32(_v16, _vs1);
                vst1q_f32(&term[_row][_col], _vr17);
                float32x4_t _v18 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v19 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr20 = vaddq_f32(_v18, _v19);
                vst1q_f32(&result[_row][_col], _vr20);
                float32x4_t _v21 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v22 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr23 = vmulq_f32(_v21, _v22);
                vst1q_f32(&term[_row][_col], _vr23);
                float32x4_t _v24 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr25 = vdivq_f32(_v24, _vs2);
                vst1q_f32(&term[_row][_col], _vr25);
                float32x4_t _v26 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v27 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr28 = vaddq_f32(_v26, _v27);
                vst1q_f32(&result[_row][_col], _vr28);
                float32x4_t _v29 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v30 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr31 = vmulq_f32(_v29, _v30);
                vst1q_f32(&term[_row][_col], _vr31);
                float32x4_t _v32 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr33 = vdivq_f32(_v32, _vs3);
                vst1q_f32(&term[_row][_col], _vr33);
                float32x4_t _v34 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v35 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr36 = vaddq_f32(_v34, _v35);
                vst1q_f32(&result[_row][_col], _vr36);
                float32x4_t _v37 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v38 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr39 = vmulq_f32(_v37, _v38);
                vst1q_f32(&term[_row][_col], _vr39);
                float32x4_t _v40 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr41 = vdivq_f32(_v40, _vs4);
                vst1q_f32(&term[_row][_col], _vr41);
                float32x4_t _v42 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v43 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr44 = vaddq_f32(_v42, _v43);
                vst1q_f32(&result[_row][_col], _vr44);
                float32x4_t _vs45 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(tile_idx) * 4096 + _row * 4096 + _col], _vs45);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(tile_idx) * 4096 + _row * 4096 + _col];
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
                output[(tile_idx) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

    int has_tail = (tail_elements > zero) ? 1 : 0;

    if (has_tail) {

        // FUSED LOOP (17 ops): x=TLOAD(input,num_full_tiles,0); result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); output=TSTORE(result,num_full_tiles,0)
        float32x4_t _vs46 = vdupq_n_f32(1.0f);
        float32x4_t _vs47 = vdupq_n_f32(6.0f);
        float32x4_t _vs48 = vdupq_n_f32(20.0f);
        float32x4_t _vs49 = vdupq_n_f32(42.0f);
        float32x4_t _vs50 = vdupq_n_f32(72.0f);
        for (int _row = 0; _row < 1; _row++) {
            int _col;
            // Vectorized loop
            for (_col = 0; _col + 4 <= 4096; _col += 4) {
                float32x4_t _vl51 = vld1q_f32(&input[(num_full_tiles) * 4096 + _row * 4096 + _col]);
                vst1q_f32(&x[_row][_col], _vl51);
                float32x4_t _v52 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr53 = vmulq_f32(_v52, _vs46);
                vst1q_f32(&result[_row][_col], _vr53);
                float32x4_t _v54 = vld1q_f32(&x[_row][_col]);
                float32x4_t _v55 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr56 = vmulq_f32(_v54, _v55);
                vst1q_f32(&x_squared[_row][_col], _vr56);
                float32x4_t _v57 = vld1q_f32(&x[_row][_col]);
                float32x4_t _vr58 = vmulq_f32(_v57, _vs46);
                vst1q_f32(&term[_row][_col], _vr58);
                float32x4_t _v59 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v60 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr61 = vmulq_f32(_v59, _v60);
                vst1q_f32(&term[_row][_col], _vr61);
                float32x4_t _v62 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr63 = vdivq_f32(_v62, _vs47);
                vst1q_f32(&term[_row][_col], _vr63);
                float32x4_t _v64 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v65 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr66 = vaddq_f32(_v64, _v65);
                vst1q_f32(&result[_row][_col], _vr66);
                float32x4_t _v67 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v68 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr69 = vmulq_f32(_v67, _v68);
                vst1q_f32(&term[_row][_col], _vr69);
                float32x4_t _v70 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr71 = vdivq_f32(_v70, _vs48);
                vst1q_f32(&term[_row][_col], _vr71);
                float32x4_t _v72 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v73 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr74 = vaddq_f32(_v72, _v73);
                vst1q_f32(&result[_row][_col], _vr74);
                float32x4_t _v75 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v76 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr77 = vmulq_f32(_v75, _v76);
                vst1q_f32(&term[_row][_col], _vr77);
                float32x4_t _v78 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr79 = vdivq_f32(_v78, _vs49);
                vst1q_f32(&term[_row][_col], _vr79);
                float32x4_t _v80 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v81 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr82 = vaddq_f32(_v80, _v81);
                vst1q_f32(&result[_row][_col], _vr82);
                float32x4_t _v83 = vld1q_f32(&term[_row][_col]);
                float32x4_t _v84 = vld1q_f32(&x_squared[_row][_col]);
                float32x4_t _vr85 = vmulq_f32(_v83, _v84);
                vst1q_f32(&term[_row][_col], _vr85);
                float32x4_t _v86 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr87 = vdivq_f32(_v86, _vs50);
                vst1q_f32(&term[_row][_col], _vr87);
                float32x4_t _v88 = vld1q_f32(&result[_row][_col]);
                float32x4_t _v89 = vld1q_f32(&term[_row][_col]);
                float32x4_t _vr90 = vaddq_f32(_v88, _v89);
                vst1q_f32(&result[_row][_col], _vr90);
                float32x4_t _vs91 = vld1q_f32(&result[_row][_col]);
                vst1q_f32(&output[(num_full_tiles) * 4096 + _row * 4096 + _col], _vs91);
            }
            // Scalar cleanup
            for (; _col < 4096; _col++) {
                x[_row][_col] = input[(num_full_tiles) * 4096 + _row * 4096 + _col];
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
                output[(num_full_tiles) * 4096 + _row * 4096 + _col] = result[_row][_col];
            }
        }

    }

}