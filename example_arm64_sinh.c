// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

/*
* PTO ISA Example: sinh() using Taylor Expansion
*
* This file demonstrates the PTO DSL for computing sinh(x) on tiles.
*
* Taylor expansion for sinh(x):
*   sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
*           = x + x³/6 + x⁵/120 + x⁷/5040 + ...
*
* Algorithm:
*   result = x
*   term = x
*   x_squared = x * x
*   for n = 1 to N:
*       term = term * x_squared / ((2n)(2n+1))
*       result = result + term
*
* PTO Instructions Used:
*   - PTO_TILE: Declare a 2D tile
*   - PTO_TMUL: Elementwise multiplication
*   - PTO_TDIVS: Elementwise division by scalar
*   - PTO_TADD: Elementwise addition
*   - PTO_TEXPANDS: Broadcast scalar to tile
*/

// ============================================================================
// PTO Tile and Scalar Declarations
// ============================================================================

// Input tile x: 8x8 tile of float32
float x[8][8];

// Working tiles
float x_squared[8][8];
float term[8][8];
float result[8][8];
float temp[8][8];

// ============================================================================
// PTO Function: compute_sinh
// Computes sinh(x) elementwise for an 8x8 tile
// Uses 7 terms of Taylor expansion for good accuracy
// ============================================================================

void compute_sinh(float* input, float* output) {
    
    // Load input tile from memory
    
    // Initialize: result = x (first term of Taylor series)
    
    // Compute x² for use in all iterations
    
    // Initialize term = x (first term)
    
    // ========================================================================
    // Taylor expansion iterations
    // term_n = term_{n-1} * x² / ((2n)(2n+1))
    // ========================================================================
    
    // Term 2: x³/3! = x³/6
    // term = term * x² = x³
    // term = term / 6
    
    // Term 3: x⁵/5! = x⁵/120
    // term = term * x² / (4*5) = (x³/6) * x² / 20
    
    // Term 4: x⁷/7! = x⁷/5040
    // term = term * x² / (6*7) = (x⁵/120) * x² / 42
    
    // Term 5: x⁹/9! = x⁹/362880
    // term = term * x² / (8*9) = (x⁷/5040) * x² / 72
    
    // Term 6: x¹¹/11! 
    // term = term * x² / (10*11) = term * x² / 110
    
    // Term 7: x¹³/13!
    // term = term * x² / (12*13) = term * x² / 156
    
    // Store result to output memory
    // Loop fusion: 21 loop overheads saved
// TLOAD: x = tload input[0, 0]
for (int _row = 0; _row < 8; _row++) {
    for (int _col = 0; _col < 8; _col++) {
        x[_row][_col] = input[(_row + 0) * 8 + (_col + 0)];
    }
}
// FUSED LOOP (22 ops): result=TADD(x,x); result=TMULS(x,1.0f); x_squared=TMUL(x,x); term=TMULS(x,1.0f); term=TMUL(term,x_squared); term=TDIVS(term,6.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,20.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,42.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,72.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,110.0f); result=TADD(result,term); term=TMUL(term,x_squared); term=TDIVS(term,156.0f); result=TADD(result,term)
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
        float32x4_t _v8 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr9 = vaddq_f32(_v7, _v8);
        vst1q_f32(&result[_row][_col], _vr9);
        float32x4_t _v10 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr11 = vmulq_f32(_v10, _vs0);
        vst1q_f32(&result[_row][_col], _vr11);
        float32x4_t _v12 = vld1q_f32(&x[_row][_col]);
        float32x4_t _v13 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr14 = vmulq_f32(_v12, _v13);
        vst1q_f32(&x_squared[_row][_col], _vr14);
        float32x4_t _v15 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr16 = vmulq_f32(_v15, _vs0);
        vst1q_f32(&term[_row][_col], _vr16);
        float32x4_t _v17 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v18 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr19 = vmulq_f32(_v17, _v18);
        vst1q_f32(&term[_row][_col], _vr19);
        float32x4_t _v20 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr21 = vdivq_f32(_v20, _vs1);
        vst1q_f32(&term[_row][_col], _vr21);
        float32x4_t _v22 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v23 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr24 = vaddq_f32(_v22, _v23);
        vst1q_f32(&result[_row][_col], _vr24);
        float32x4_t _v25 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v26 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr27 = vmulq_f32(_v25, _v26);
        vst1q_f32(&term[_row][_col], _vr27);
        float32x4_t _v28 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr29 = vdivq_f32(_v28, _vs2);
        vst1q_f32(&term[_row][_col], _vr29);
        float32x4_t _v30 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v31 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr32 = vaddq_f32(_v30, _v31);
        vst1q_f32(&result[_row][_col], _vr32);
        float32x4_t _v33 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v34 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr35 = vmulq_f32(_v33, _v34);
        vst1q_f32(&term[_row][_col], _vr35);
        float32x4_t _v36 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr37 = vdivq_f32(_v36, _vs3);
        vst1q_f32(&term[_row][_col], _vr37);
        float32x4_t _v38 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v39 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr40 = vaddq_f32(_v38, _v39);
        vst1q_f32(&result[_row][_col], _vr40);
        float32x4_t _v41 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v42 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr43 = vmulq_f32(_v41, _v42);
        vst1q_f32(&term[_row][_col], _vr43);
        float32x4_t _v44 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr45 = vdivq_f32(_v44, _vs4);
        vst1q_f32(&term[_row][_col], _vr45);
        float32x4_t _v46 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v47 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr48 = vaddq_f32(_v46, _v47);
        vst1q_f32(&result[_row][_col], _vr48);
        float32x4_t _v49 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v50 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr51 = vmulq_f32(_v49, _v50);
        vst1q_f32(&term[_row][_col], _vr51);
        float32x4_t _v52 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr53 = vdivq_f32(_v52, _vs5);
        vst1q_f32(&term[_row][_col], _vr53);
        float32x4_t _v54 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v55 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr56 = vaddq_f32(_v54, _v55);
        vst1q_f32(&result[_row][_col], _vr56);
        float32x4_t _v57 = vld1q_f32(&term[_row][_col]);
        float32x4_t _v58 = vld1q_f32(&x_squared[_row][_col]);
        float32x4_t _vr59 = vmulq_f32(_v57, _v58);
        vst1q_f32(&term[_row][_col], _vr59);
        float32x4_t _v60 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr61 = vdivq_f32(_v60, _vs6);
        vst1q_f32(&term[_row][_col], _vr61);
        float32x4_t _v62 = vld1q_f32(&result[_row][_col]);
        float32x4_t _v63 = vld1q_f32(&term[_row][_col]);
        float32x4_t _vr64 = vaddq_f32(_v62, _v63);
        vst1q_f32(&result[_row][_col], _vr64);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = x[_row][_col] + x[_row][_col];
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
}

// ============================================================================
// Main function for testing
// ============================================================================

int main() {
    // Allocate input and output arrays
float input[8][8];
float output[8][8];
    
    // Initialize input with test values
for (int i = 0; i < 8; i++) {
for (int j = 0; j < 8; j++) {
            // Values from -2 to 2 for sinh test
input[i][j] = -2.0f + (i * 8 + j) * (4.0f / 64.0f);
}
}
    
    // Call PTO sinh function
compute_sinh((float*)input, (float*)output);
    
    // Print results and compare with standard sinh
printf("PTO sinh() Test Results:\n");
printf("========================\n\n");
    
float max_error = 0.0f;
for (int i = 0; i < 8; i++) {
for (int j = 0; j < 8; j++) {
float x_val = input[i][j];
float pto_result = output[i][j];
float expected = sinhf(x_val);
float error = fabsf(pto_result - expected);
if (error > max_error) max_error = error;
            
if (i < 2 && j < 4) {  // Print first few results
printf("sinh(%6.3f) = %10.6f (expected: %10.6f, error: %.6f)\n",
x_val, pto_result, expected, error);
}
}
}
    
printf("\n...\n");
printf("\nMax error: %.9f\n", max_error);
printf("Test %s\n", max_error < 1e-4 ? "PASSED" : "FAILED");
    
return 0;
}
