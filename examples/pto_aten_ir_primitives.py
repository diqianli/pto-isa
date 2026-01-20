"""
PTO ATen IR Primitives Implementation with Dynamic Tiling

This module implements ATen IR primitives using PTO ISA instructions with
dynamic tensor shape support.

Based on PyTorch's Core Aten IR and Prims IR:
https://docs.pytorch.org/docs/stable/torch.compiler_ir.html

Tile Shape Computation Rules:
1) col should be multiples of VECTOR_LANES of the given physical ISA
2) row should be multiple of PHYSICAL_ROW_SIZE
3) byte size of the TILE should be no greater than 16KB

Categories implemented:
1. Elementwise Unary: abs, neg, exp, log, sqrt, rsqrt, reciprocal, sigmoid, tanh, relu
2. Elementwise Binary: add, sub, mul, div, maximum, minimum, pow
3. Scalar Operations: add_scalar, mul_scalar, div_scalar
4. Reductions: sum, mean, amax, amin
5. Composite Operations: softmax, gelu, silu, sinh, cosh
"""

import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pto_compile import PTOFunctionBuilder, PTOCompiler
from pto_isa_definition import ElementType, MemorySpace, CompareMode

# Import dynamic tiling utilities (from parent directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pto_dynamic_tiling import (
    compute_tile_shape, get_tile_info, 
    build_unary_op, build_binary_op, build_scalar_op,
    DEFAULT_DTYPE, MAX_TILE_BYTES, ELEMENT_BYTES
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TARGET_ISA = "arm64"


# =============================================================================
# Helper: Build Dynamic Tiled Program
# =============================================================================

def _build_dynamic_program(name, dtype, target_isa, build_computation):
    """
    Helper to build a dynamic tiled program.
    
    Args:
        name: Program name
        dtype: Element data type
        target_isa: Target ISA
        build_computation: Function that takes (builder, offset_var) and adds operations
    
    Returns:
        PTOProgram
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    return build_computation(rows, cols, tile_elements)


# =============================================================================
# Elementwise Unary Primitives with Dynamic Tiling
# =============================================================================

def prims_abs(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.abs: (Tensor self) -> Tensor - Elementwise absolute value."""
    return build_unary_op("prims_abs", "abs", dtype, target_isa)


def prims_neg(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.neg: (Tensor self) -> Tensor - Elementwise negation."""
    return build_unary_op("prims_neg", "neg", dtype, target_isa)


def prims_exp(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.exp: (Tensor self) -> Tensor - Elementwise exponential."""
    return build_unary_op("prims_exp", "exp", dtype, target_isa)


def prims_log(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.log: (Tensor self) -> Tensor - Elementwise natural logarithm."""
    return build_unary_op("prims_log", "log", dtype, target_isa)


def prims_sqrt(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.sqrt: (Tensor self) -> Tensor - Elementwise square root."""
    return build_unary_op("prims_sqrt", "sqrt", dtype, target_isa)


def prims_rsqrt(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.rsqrt: (Tensor self) -> Tensor - Elementwise reciprocal square root."""
    return build_unary_op("prims_rsqrt", "rsqrt", dtype, target_isa)


def prims_reciprocal(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.reciprocal: (Tensor self) -> Tensor - Elementwise reciprocal."""
    return build_unary_op("prims_reciprocal", "recip", dtype, target_isa)


def aten_relu(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """aten.relu: (Tensor self) -> Tensor - Elementwise ReLU."""
    return build_unary_op("aten_relu", "relu", dtype, target_isa)


# =============================================================================
# Composite Unary Operations with Dynamic Tiling
# =============================================================================

def aten_sigmoid(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """
    aten.sigmoid: (Tensor self) -> Tensor
    Elementwise sigmoid: 1 / (1 + exp(-x))
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("aten_sigmoid")
        .tile("x", rows, cols, dtype)
        .tile("t1", rows, cols, dtype)
        .tile("t2", rows, cols, dtype)
        .tile("t3", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        # Main loop
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .neg("t1", "x")
            .exp("t2", "t1")
            .adds("t3", "t2", 1.0)
            .recip("result", "t3")
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        # Tail handling
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .neg("t1", "x")
            .exp("t2", "t1")
            .adds("t3", "t2", 1.0)
            .recip("result", "t3")
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


def aten_tanh(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """
    aten.tanh: (Tensor self) -> Tensor
    Elementwise hyperbolic tangent: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("aten_tanh")
        .tile("x", rows, cols, dtype)
        .tile("exp_x", rows, cols, dtype)
        .tile("exp_neg_x", rows, cols, dtype)
        .tile("neg_x", rows, cols, dtype)
        .tile("numerator", rows, cols, dtype)
        .tile("denominator", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .exp("exp_x", "x")
            .neg("neg_x", "x")
            .exp("exp_neg_x", "neg_x")
            .sub("numerator", "exp_x", "exp_neg_x")
            .add("denominator", "exp_x", "exp_neg_x")
            .div("result", "numerator", "denominator")
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .exp("exp_x", "x")
            .neg("neg_x", "x")
            .exp("exp_neg_x", "neg_x")
            .sub("numerator", "exp_x", "exp_neg_x")
            .add("denominator", "exp_x", "exp_neg_x")
            .div("result", "numerator", "denominator")
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


# =============================================================================
# Elementwise Binary Primitives with Dynamic Tiling
# =============================================================================

def prims_add(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.add: (Tensor self, Tensor other) -> Tensor - Elementwise addition."""
    return build_binary_op("prims_add", "add", dtype, target_isa)


def prims_sub(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.sub: (Tensor self, Tensor other) -> Tensor - Elementwise subtraction."""
    return build_binary_op("prims_sub", "sub", dtype, target_isa)


def prims_mul(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.mul: (Tensor self, Tensor other) -> Tensor - Elementwise multiplication."""
    return build_binary_op("prims_mul", "mul", dtype, target_isa)


def prims_div(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.div: (Tensor self, Tensor other) -> Tensor - Elementwise division."""
    return build_binary_op("prims_div", "div", dtype, target_isa)


def prims_maximum(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.maximum: (Tensor self, Tensor other) -> Tensor - Elementwise maximum."""
    return build_binary_op("prims_maximum", "max", dtype, target_isa)


def prims_minimum(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.minimum: (Tensor self, Tensor other) -> Tensor - Elementwise minimum."""
    return build_binary_op("prims_minimum", "min", dtype, target_isa)


def prims_pow(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """
    prims.pow: (Tensor base, Tensor exponent) -> Tensor
    Elementwise power: base^exponent = exp(exponent * log(base))
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("prims_pow")
        .tile("base", rows, cols, dtype)
        .tile("exp_tile", rows, cols, dtype)
        .tile("log_base", rows, cols, dtype)
        .tile("product", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_base", MemorySpace.GM, dtype)
        .memref("input_exp", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("base", "input_base", "tile_idx", 0)
            .load("exp_tile", "input_exp", "tile_idx", 0)
            .log("log_base", "base")
            .mul("product", "exp_tile", "log_base")
            .exp("result", "product")
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("base", "input_base", "num_full_tiles", 0)
            .load("exp_tile", "input_exp", "num_full_tiles", 0)
            .log("log_base", "base")
            .mul("product", "exp_tile", "log_base")
            .exp("result", "product")
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


# =============================================================================
# Scalar Operations with Dynamic Tiling
# =============================================================================

def aten_add_scalar(scalar_value, dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """aten.add.Scalar: (Tensor self, Scalar other) -> Tensor"""
    return build_scalar_op("aten_add_scalar", "adds", scalar_value, dtype, target_isa)


def aten_mul_scalar(scalar_value, dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """aten.mul.Scalar: (Tensor self, Scalar other) -> Tensor"""
    return build_scalar_op("aten_mul_scalar", "muls", scalar_value, dtype, target_isa)


def aten_div_scalar(scalar_value, dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """aten.div.Scalar: (Tensor self, Scalar other) -> Tensor"""
    return build_scalar_op("aten_div_scalar", "divs", scalar_value, dtype, target_isa)


# =============================================================================
# Reduction Operations with Dynamic Tiling
# =============================================================================

def prims_sum_row(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """prims.sum: (Tensor inp, int[] dims) -> Tensor - Sum reduction along rows."""
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("prims_sum_row")
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .rowsum("result", "x")
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .rowsum("result", "x")
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


def aten_mean_row(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """aten.mean: (Tensor self, int[] dim) -> Tensor - Mean reduction along rows."""
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("aten_mean_row")
        .tile("x", rows, cols, dtype)
        .tile("sum_result", rows, 1, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .rowsum("sum_result", "x")
            .divs("result", "sum_result", float(cols))
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .rowsum("sum_result", "x")
            .divs("result", "sum_result", float(cols))
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


# =============================================================================
# Matrix Operations (Fixed Size - Not Dynamically Tiled)
# =============================================================================

def aten_mm(m=8, k=8, n=8, dtype=DEFAULT_DTYPE):
    """
    aten.mm: (Tensor self, Tensor mat2) -> Tensor
    Matrix multiplication: result = self @ mat2
    Note: Matrix operations use fixed tile sizes for now.
    """
    return (PTOFunctionBuilder("aten_mm")
        .tile("a", m, k, dtype)
        .tile("b", k, n, dtype)
        .tile("result", m, n, dtype)
        .memref("input_a", MemorySpace.GM, dtype)
        .memref("input_b", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        .load("a", "input_a", 0, 0)
        .load("b", "input_b", 0, 0)
        .matmul("result", "a", "b")
        .store("result", "output", 0, 0)
        
        .build())


# =============================================================================
# Activation Functions with Dynamic Tiling
# =============================================================================

def aten_gelu(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """
    aten.gelu: (Tensor self) -> Tensor
    GELU approximation: x * sigmoid(1.702 * x)
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("aten_gelu")
        .tile("x", rows, cols, dtype)
        .tile("scaled_x", rows, cols, dtype)
        .tile("neg_scaled", rows, cols, dtype)
        .tile("exp_neg", rows, cols, dtype)
        .tile("one_plus", rows, cols, dtype)
        .tile("sigmoid_out", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .muls("scaled_x", "x", 1.702)
            .neg("neg_scaled", "scaled_x")
            .exp("exp_neg", "neg_scaled")
            .adds("one_plus", "exp_neg", 1.0)
            .recip("sigmoid_out", "one_plus")
            .mul("result", "x", "sigmoid_out")
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .muls("scaled_x", "x", 1.702)
            .neg("neg_scaled", "scaled_x")
            .exp("exp_neg", "neg_scaled")
            .adds("one_plus", "exp_neg", 1.0)
            .recip("sigmoid_out", "one_plus")
            .mul("result", "x", "sigmoid_out")
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


def aten_silu(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """
    aten.silu (Swish): (Tensor self) -> Tensor
    SiLU: x * sigmoid(x)
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("aten_silu")
        .tile("x", rows, cols, dtype)
        .tile("neg_x", rows, cols, dtype)
        .tile("exp_neg", rows, cols, dtype)
        .tile("one_plus", rows, cols, dtype)
        .tile("sigmoid_out", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .neg("neg_x", "x")
            .exp("exp_neg", "neg_x")
            .adds("one_plus", "exp_neg", 1.0)
            .recip("sigmoid_out", "one_plus")
            .mul("result", "x", "sigmoid_out")
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .neg("neg_x", "x")
            .exp("exp_neg", "neg_x")
            .adds("one_plus", "exp_neg", 1.0)
            .recip("sigmoid_out", "one_plus")
            .mul("result", "x", "sigmoid_out")
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


# =============================================================================
# Hyperbolic Functions with Dynamic Tiling
# =============================================================================

def aten_sinh(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """
    aten.sinh: (Tensor self) -> Tensor
    Hyperbolic sine using Taylor expansion.
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("aten_sinh")
        .tile("x", rows, cols, dtype)
        .tile("x_squared", rows, cols, dtype)
        .tile("term", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .muls("result", "x", 1.0)
            .mul("x_squared", "x", "x")
            .muls("term", "x", 1.0)
            .mul("term", "term", "x_squared")
            .divs("term", "term", 6.0)
            .add("result", "result", "term")
            .mul("term", "term", "x_squared")
            .divs("term", "term", 20.0)
            .add("result", "result", "term")
            .mul("term", "term", "x_squared")
            .divs("term", "term", 42.0)
            .add("result", "result", "term")
            .mul("term", "term", "x_squared")
            .divs("term", "term", 72.0)
            .add("result", "result", "term")
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .muls("result", "x", 1.0)
            .mul("x_squared", "x", "x")
            .muls("term", "x", 1.0)
            .mul("term", "term", "x_squared")
            .divs("term", "term", 6.0)
            .add("result", "result", "term")
            .mul("term", "term", "x_squared")
            .divs("term", "term", 20.0)
            .add("result", "result", "term")
            .mul("term", "term", "x_squared")
            .divs("term", "term", 42.0)
            .add("result", "result", "term")
            .mul("term", "term", "x_squared")
            .divs("term", "term", 72.0)
            .add("result", "result", "term")
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


def aten_cosh(dtype=DEFAULT_DTYPE, target_isa=DEFAULT_TARGET_ISA):
    """
    aten.cosh: (Tensor self) -> Tensor
    Hyperbolic cosine: cosh(x) = (exp(x) + exp(-x)) / 2
    """
    rows, cols = compute_tile_shape(dtype, target_isa)
    tile_elements = rows * cols
    
    return (PTOFunctionBuilder("aten_cosh")
        .tile("x", rows, cols, dtype)
        .tile("neg_x", rows, cols, dtype)
        .tile("exp_x", rows, cols, dtype)
        .tile("exp_neg_x", rows, cols, dtype)
        .tile("sum", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .scalar("num_full_tiles", ElementType.I32)
        .scalar("tail_elements", ElementType.I32)
        .scalar("has_tail", ElementType.U1)
        .scalar("zero", ElementType.I32)
        .scalar("tile_size", ElementType.I32)
        .scalar_li("tile_size", tile_elements)
        .scalar_li("zero", 0)
        
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            .load("x", "input", "tile_idx", 0)
            .exp("exp_x", "x")
            .neg("neg_x", "x")
            .exp("exp_neg_x", "neg_x")
            .add("sum", "exp_x", "exp_neg_x")
            .divs("result", "sum", 2.0)
            .store("result", "output", "tile_idx", 0)
        .end_for()
        
        .scalar_cmp("has_tail", "tail_elements", "zero", CompareMode.GT)
        .if_then("has_tail")
            .load("x", "input", "num_full_tiles", 0)
            .exp("exp_x", "x")
            .neg("neg_x", "x")
            .exp("exp_neg_x", "neg_x")
            .add("sum", "exp_x", "exp_neg_x")
            .divs("result", "sum", 2.0)
            .store("result", "output", "num_full_tiles", 0)
        .endif()
        
        .build())


# =============================================================================
# Registry of All Primitives
# =============================================================================

PRIMITIVE_REGISTRY = {
    # Elementwise unary
    "prims.abs": prims_abs,
    "prims.neg": prims_neg,
    "prims.exp": prims_exp,
    "prims.log": prims_log,
    "prims.sqrt": prims_sqrt,
    "prims.rsqrt": prims_rsqrt,
    "prims.reciprocal": prims_reciprocal,
    "aten.relu": aten_relu,
    "aten.sigmoid": aten_sigmoid,
    "aten.tanh": aten_tanh,
    
    # Elementwise binary
    "prims.add": prims_add,
    "prims.sub": prims_sub,
    "prims.mul": prims_mul,
    "prims.div": prims_div,
    "prims.maximum": prims_maximum,
    "prims.minimum": prims_minimum,
    "prims.pow": prims_pow,
    
    # Scalar operations
    "aten.add.Scalar": aten_add_scalar,
    "aten.mul.Scalar": aten_mul_scalar,
    "aten.div.Scalar": aten_div_scalar,
    
    # Reductions
    "prims.sum_row": prims_sum_row,
    "aten.mean_row": aten_mean_row,
    
    # Matrix operations
    "aten.mm": aten_mm,
    
    # Activations
    "aten.gelu": aten_gelu,
    "aten.silu": aten_silu,
    
    # Hyperbolic
    "aten.sinh": aten_sinh,
    "aten.cosh": aten_cosh,
}


# =============================================================================
# Main: Generate All Primitives for All Backends
# =============================================================================

if __name__ == "__main__":
    from pto_compile import generate_all_backends, BACKENDS
    from pto_dynamic_tiling import print_tile_shapes
    
    print("=" * 70)
    print("PTO ATen IR Primitives - Dynamic Tiling Multi-Backend Code Generation")
    print("=" * 70)
    
    # Show tile shapes
    print_tile_shapes()
    
    # Base output directory
    OUTPUT_PREFIX = "aten_primitives"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    compiler = PTOCompiler()
    
    # Generate code for all primitives
    primitives_to_generate = [
        # Basic unary
        ("prims.abs", [], {}),
        ("prims.neg", [], {}),
        ("prims.exp", [], {}),
        ("prims.log", [], {}),
        ("prims.sqrt", [], {}),
        ("prims.rsqrt", [], {}),
        ("prims.reciprocal", [], {}),
        ("aten.relu", [], {}),
        
        # Composite unary
        ("aten.sigmoid", [], {}),
        ("aten.tanh", [], {}),
        
        # Binary
        ("prims.add", [], {}),
        ("prims.sub", [], {}),
        ("prims.mul", [], {}),
        ("prims.div", [], {}),
        ("prims.maximum", [], {}),
        ("prims.minimum", [], {}),
        ("prims.pow", [], {}),
        
        # Scalar
        ("aten.add.Scalar", [2.0], {}),
        ("aten.mul.Scalar", [0.5], {}),
        ("aten.div.Scalar", [4.0], {}),
        
        # Activations
        ("aten.gelu", [], {}),
        ("aten.silu", [], {}),
        
        # Hyperbolic
        ("aten.sinh", [], {}),
        ("aten.cosh", [], {}),
        
        # Matrix
        ("aten.mm", [], {'m': 8, 'k': 8, 'n': 8}),
        
        # Reduction
        ("prims.sum_row", [], {}),
        ("aten.mean_row", [], {}),
    ]
    
    print(f"\nGenerating {len(primitives_to_generate)} primitives for {len(BACKENDS)} backends...")
    print(f"Backends: {', '.join(BACKENDS.keys())}")
    print()
    
    total_files = 0
    
    for prim_name, args, kwargs in primitives_to_generate:
        safe_name = prim_name.replace(".", "_").replace("Scalar", "scalar")
        
        try:
            builder_func = PRIMITIVE_REGISTRY[prim_name]
            if prim_name in ("aten.add.Scalar", "aten.mul.Scalar", "aten.div.Scalar"):
                program = builder_func(args[0])
            elif prim_name == "aten.mm":
                program = builder_func(**kwargs)
            else:
                program = builder_func()
            
            print(f"[{prim_name}]")
            
            results = generate_all_backends(
                program, 
                OUTPUT_PREFIX,
                output_base_dir=SCRIPT_DIR,
                enable_fusion=True
            )
            
            total_files += len(results)
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'=' * 70}")
    print("Generation Complete!")
    print(f"Total files generated: {total_files}")
    print(f"Output directories:")
    for backend_key, backend_info in BACKENDS.items():
        print(f"  - output{backend_info['suffix']}/{OUTPUT_PREFIX}/")
    print(f"  - output_pto/{OUTPUT_PREFIX}/")
    print("=" * 70)
