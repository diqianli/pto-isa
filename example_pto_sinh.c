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
PTO_TILE(x, 8, 8, f32)

// Working tiles
PTO_TILE(x_squared, 8, 8, f32)      // x²
PTO_TILE(term, 8, 8, f32)           // Current Taylor term
PTO_TILE(result, 8, 8, f32)         // Accumulated result
PTO_TILE(temp, 8, 8, f32)           // Temporary for computations

// ============================================================================
// PTO Function: compute_sinh
// Computes sinh(x) elementwise for an 8x8 tile
// Uses 7 terms of Taylor expansion for good accuracy
// ============================================================================

PTO_FUNCTION_START
void compute_sinh(float* input, float* output) {
    
    // Load input tile from memory
    PTO_TLOAD(x, input, 0, 0)
    
    // Initialize: result = x (first term of Taylor series)
    PTO_TADD(result, x, x)           // result = x + x = 2x (temporary)
    PTO_TMULS(result, x, 1.0f)       // result = x * 1.0 = x (copy x to result)
    
    // Compute x² for use in all iterations
    PTO_TMUL(x_squared, x, x)        // x_squared = x * x
    
    // Initialize term = x (first term)
    PTO_TMULS(term, x, 1.0f)         // term = x
    
    // ========================================================================
    // Taylor expansion iterations
    // term_n = term_{n-1} * x² / ((2n)(2n+1))
    // ========================================================================
    
    // Term 2: x³/3! = x³/6
    // term = term * x² = x³
    // term = term / 6
    PTO_TMUL(term, term, x_squared)  // term = x * x² = x³
    PTO_TDIVS(term, term, 6.0f)      // term = x³/6
    PTO_TADD(result, result, term)   // result = x + x³/6
    
    // Term 3: x⁵/5! = x⁵/120
    // term = term * x² / (4*5) = (x³/6) * x² / 20
    PTO_TMUL(term, term, x_squared)  // term = x⁵/6
    PTO_TDIVS(term, term, 20.0f)     // term = x⁵/120
    PTO_TADD(result, result, term)   // result = x + x³/6 + x⁵/120
    
    // Term 4: x⁷/7! = x⁷/5040
    // term = term * x² / (6*7) = (x⁵/120) * x² / 42
    PTO_TMUL(term, term, x_squared)  // term = x⁷/120
    PTO_TDIVS(term, term, 42.0f)     // term = x⁷/5040
    PTO_TADD(result, result, term)   // result += x⁷/5040
    
    // Term 5: x⁹/9! = x⁹/362880
    // term = term * x² / (8*9) = (x⁷/5040) * x² / 72
    PTO_TMUL(term, term, x_squared)  // term = x⁹/5040
    PTO_TDIVS(term, term, 72.0f)     // term = x⁹/362880
    PTO_TADD(result, result, term)   // result += x⁹/362880
    
    // Term 6: x¹¹/11! 
    // term = term * x² / (10*11) = term * x² / 110
    PTO_TMUL(term, term, x_squared)
    PTO_TDIVS(term, term, 110.0f)
    PTO_TADD(result, result, term)
    
    // Term 7: x¹³/13!
    // term = term * x² / (12*13) = term * x² / 156
    PTO_TMUL(term, term, x_squared)
    PTO_TDIVS(term, term, 156.0f)
    PTO_TADD(result, result, term)
    
    // Store result to output memory
    PTO_TSTORE(result, output, 0, 0)
}
PTO_FUNCTION_END

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
