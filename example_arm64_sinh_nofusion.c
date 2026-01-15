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
// TLOAD: x = tload input[0, 0]
for (int _row = 0; _row < 8; _row++) {
    for (int _col = 0; _col < 8; _col++) {
        x[_row][_col] = input[(_row + 0) * 8 + (_col + 0)];
    }
}
    
    // Initialize: result = x (first term of Taylor series)
// TADD: result = tadd x, x
for (int _row = 0; _row < 8; _row++) {
    int _col;
    // Vectorized loop
    for (_col = 0; _col + 4 <= 8; _col += 4) {
        float32x4_t _v0 = vld1q_f32(&x[_row][_col]);
        float32x4_t _v1 = vld1q_f32(&x[_row][_col]);
        float32x4_t _vr = vaddq_f32(_v0, _v1);
        vst1q_f32(&result[_row][_col], _vr);
    }
    // Scalar cleanup
    for (; _col < 8; _col++) {
        result[_row][_col] = x[_row][_col] + x[_row][_col];
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
    
    // Compute x² for use in all iterations
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
    
    // Initialize term = x (first term)
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
    
    // ========================================================================
    // Taylor expansion iterations
    // term_n = term_{n-1} * x² / ((2n)(2n+1))
    // ========================================================================
    
    // Term 2: x³/3! = x³/6
    // term = term * x² = x³
    // term = term / 6
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
    
    // Term 3: x⁵/5! = x⁵/120
    // term = term * x² / (4*5) = (x³/6) * x² / 20
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
    
    // Term 4: x⁷/7! = x⁷/5040
    // term = term * x² / (6*7) = (x⁵/120) * x² / 42
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
    
    // Term 5: x⁹/9! = x⁹/362880
    // term = term * x² / (8*9) = (x⁷/5040) * x² / 72
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
    
    // Term 6: x¹¹/11! 
    // term = term * x² / (10*11) = term * x² / 110
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
    
    // Term 7: x¹³/13!
    // term = term * x² / (12*13) = term * x² / 156
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
    
    // Store result to output memory
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
