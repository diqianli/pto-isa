"""
ARM64 Code Generation for PTO ISA (Legacy Module)

NOTE: This module is maintained for backward compatibility.
The preferred approach is to use the codegen_arm64() method on each PTO instruction class
in pto_isa_definition.py. This allows:
  - Consistent instruction definition with codegen in one place
  - Easy addition of new backends (e.g., codegen_riscv(), codegen_x86())
  - The same compiler can target different architectures by calling the appropriate method

For new code, use:
    from pto_isa_definition import TADD, ARM64CodeGenContext
    ctx = ARM64CodeGenContext()
    instr = TADD(dst, src0, src1)
    lines = instr.codegen_arm64(ctx)

This module provides code generation from PTO instructions to ARM64 NEON/SVE intrinsics.
Tiles are mapped to 2D arrays with vectorized inner loops using NEON intrinsics.

ARM64 NEON Intrinsics Reference:
- float32x4_t: 128-bit vector holding 4 x float32
- float16x8_t: 128-bit vector holding 8 x float16
- vld1q_f32/vst1q_f32: Load/store 4 floats
- vaddq_f32, vsubq_f32, vmulq_f32, vdivq_f32: Arithmetic
- vmaxq_f32, vminq_f32: Min/max
- vabsq_f32, vnegq_f32: Unary ops
- vdupq_n_f32: Broadcast scalar
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# ARM64 Type Mappings
# =============================================================================

ARM64_TYPE_MAP = {
    "f32": "float",
    "f16": "__fp16",
    "f64": "double",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}

ARM64_VECTOR_TYPE_MAP = {
    "f32": "float32x4_t",
    "f16": "float16x8_t",
    "f64": "float64x2_t",
    "i8": "int8x16_t",
    "i16": "int16x8_t",
    "i32": "int32x4_t",
    "i64": "int64x2_t",
    "u8": "uint8x16_t",
    "u16": "uint16x8_t",
    "u32": "uint32x4_t",
    "u64": "uint64x2_t",
}

# Elements per NEON vector
VECTOR_LANES = {
    "f32": 4,
    "f16": 8,
    "f64": 2,
    "i8": 16,
    "i16": 8,
    "i32": 4,
    "i64": 2,
    "u8": 16,
    "u16": 8,
    "u32": 4,
    "u64": 2,
}


# =============================================================================
# Code Generation Context
# =============================================================================

@dataclass
class CodeGenContext:
    """Context for code generation."""
    indent_level: int = 0
    temp_counter: int = 0
    declared_vars: set = None
    var_counter: int = 0
    
    def __post_init__(self):
        if self.declared_vars is None:
            self.declared_vars = set()
    
    def get_temp(self, prefix: str = "tmp") -> str:
        """Get a unique temporary variable name."""
        name = f"{prefix}_{self.temp_counter}"
        self.temp_counter += 1
        return name
    
    def get_unique_var(self, prefix: str = "_v") -> str:
        """Get a unique variable name."""
        name = f"{prefix}{self.var_counter}"
        self.var_counter += 1
        return name
    
    def indent(self) -> str:
        """Get current indentation string."""
        return "    " * self.indent_level


# =============================================================================
# ARM64 Intrinsic Code Generators
# =============================================================================

class ARM64CodeGen:
    """
    ARM64 NEON intrinsic code generator for PTO instructions.
    """
    
    def __init__(self):
        self.ctx = CodeGenContext()
    
    def reset(self):
        """Reset the code generator state."""
        self.ctx = CodeGenContext()
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _get_neon_suffix(self, dtype: str) -> str:
        """Get NEON intrinsic suffix for data type."""
        suffix_map = {
            "f32": "f32",
            "f16": "f16",
            "f64": "f64",
            "i8": "s8",
            "i16": "s16",
            "i32": "s32",
            "i64": "s64",
            "u8": "u8",
            "u16": "u16",
            "u32": "u32",
            "u64": "u64",
        }
        return suffix_map.get(dtype, "f32")
    
    def _emit_tile_loop(self, rows: int, cols: int, dtype: str, 
                        body_gen, dst_name: str, *src_names) -> List[str]:
        """
        Generate vectorized tile loop with NEON intrinsics.
        
        Args:
            rows, cols: Tile dimensions
            dtype: Element data type
            body_gen: Function to generate loop body
            dst_name: Destination tile name
            src_names: Source tile names
        """
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        suffix = self._get_neon_suffix(dtype)
        
        # Generate loop over rows
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        self.ctx.indent_level += 1
        indent = self.ctx.indent()
        
        # Vectorized loop over columns
        lines.append(f"{indent}int _col;")
        lines.append(f"{indent}for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        self.ctx.indent_level += 1
        
        # Generate vectorized body
        vec_body = body_gen(dtype, suffix, vec_type, dst_name, src_names, vectorized=True)
        lines.extend(vec_body)
        
        self.ctx.indent_level -= 1
        indent = self.ctx.indent()
        lines.append(f"{indent}}}")
        
        # Scalar cleanup for remaining elements
        lines.append(f"{indent}for (; _col < {cols}; _col++) {{")
        self.ctx.indent_level += 1
        
        scalar_body = body_gen(dtype, suffix, vec_type, dst_name, src_names, vectorized=False)
        lines.extend(scalar_body)
        
        self.ctx.indent_level -= 1
        indent = self.ctx.indent()
        lines.append(f"{indent}}}")
        
        self.ctx.indent_level -= 1
        indent = self.ctx.indent()
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Tile Load/Store
    # =========================================================================
    
    def gen_tload(self, dst: str, src_mem: str, row_off: int, col_off: int,
                  rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TLOAD - load tile from memory."""
        lines = []
        indent = self.ctx.indent()
        c_type = ARM64_TYPE_MAP.get(dtype, "float")
        
        lines.append(f"{indent}// TLOAD: {dst} = tload {src_mem}[{row_off}, {col_off}]")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = {src_mem}[(_row + {row_off}) * {cols} + (_col + {col_off})];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tstore(self, src: str, dst_mem: str, row_off: int, col_off: int,
                   rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TSTORE - store tile to memory."""
        lines = []
        indent = self.ctx.indent()
        
        lines.append(f"{indent}// TSTORE: tstore {src}, {dst_mem}[{row_off}, {col_off}]")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst_mem}[(_row + {row_off}) * {cols} + (_col + {col_off})] = {src}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Elementwise Binary Operations
    # =========================================================================
    
    def gen_tadd(self, dst: str, src0: str, src1: str, 
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TADD - elementwise add."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TADD: {dst} = tadd {src0}, {src1}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    // Vectorized loop")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src0}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _v1 = vld1q_{suffix}(&{src1}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vaddq_{suffix}(_v0, _v1);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    // Scalar cleanup")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = {src0}[_row][_col] + {src1}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tsub(self, dst: str, src0: str, src1: str,
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TSUB - elementwise subtract."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TSUB: {dst} = tsub {src0}, {src1}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src0}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _v1 = vld1q_{suffix}(&{src1}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vsubq_{suffix}(_v0, _v1);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = {src0}[_row][_col] - {src1}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tmul(self, dst: str, src0: str, src1: str,
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TMUL - elementwise multiply."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TMUL: {dst} = tmul {src0}, {src1}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src0}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _v1 = vld1q_{suffix}(&{src1}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vmulq_{suffix}(_v0, _v1);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = {src0}[_row][_col] * {src1}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tdiv(self, dst: str, src0: str, src1: str,
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TDIV - elementwise divide."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TDIV: {dst} = tdiv {src0}, {src1}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src0}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _v1 = vld1q_{suffix}(&{src1}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vdivq_{suffix}(_v0, _v1);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = {src0}[_row][_col] / {src1}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Elementwise Scalar Operations
    # =========================================================================
    
    def gen_tadds(self, dst: str, src: str, scalar: str,
                  rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TADDS - add scalar to tile."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        vs = self.ctx.get_unique_var("_vs")
        
        lines.append(f"{indent}// TADDS: {dst} = tadds {src}, {scalar}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {vec_type} {vs} = vdupq_n_{suffix}({scalar});")
        lines.append(f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}        int _col;")
        lines.append(f"{indent}        for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}            {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}            {vec_type} _vr = vaddq_{suffix}(_v0, {vs});")
        lines.append(f"{indent}            vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}        for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}            {dst}[_row][_col] = {src}[_row][_col] + {scalar};")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tmuls(self, dst: str, src: str, scalar: str,
                  rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TMULS - multiply tile by scalar."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        vs = self.ctx.get_unique_var("_vs")
        
        lines.append(f"{indent}// TMULS: {dst} = tmuls {src}, {scalar}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {vec_type} {vs} = vdupq_n_{suffix}({scalar});")
        lines.append(f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}        int _col;")
        lines.append(f"{indent}        for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}            {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}            {vec_type} _vr = vmulq_{suffix}(_v0, {vs});")
        lines.append(f"{indent}            vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}        for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}            {dst}[_row][_col] = {src}[_row][_col] * {scalar};")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tdivs(self, dst: str, src: str, scalar: str,
                  rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TDIVS - divide tile by scalar."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        vs = self.ctx.get_unique_var("_vs")
        
        lines.append(f"{indent}// TDIVS: {dst} = tdivs {src}, {scalar}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {vec_type} {vs} = vdupq_n_{suffix}({scalar});")
        lines.append(f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}        int _col;")
        lines.append(f"{indent}        for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}            {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}            {vec_type} _vr = vdivq_{suffix}(_v0, {vs});")
        lines.append(f"{indent}            vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}        for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}            {dst}[_row][_col] = {src}[_row][_col] / {scalar};")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Elementwise Unary Operations
    # =========================================================================
    
    def gen_tabs(self, dst: str, src: str,
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TABS - elementwise absolute value."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        c_type = ARM64_TYPE_MAP.get(dtype, "float")
        
        lines.append(f"{indent}// TABS: {dst} = tabs {src}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vabsq_{suffix}(_v0);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = fabsf({src}[_row][_col]);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tneg(self, dst: str, src: str,
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TNEG - elementwise negation."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TNEG: {dst} = tneg {src}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vnegq_{suffix}(_v0);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = -{src}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_trecip(self, dst: str, src: str,
                   rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TRECIP - elementwise reciprocal."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        vone = self.ctx.get_unique_var("_vone")
        
        lines.append(f"{indent}// TRECIP: {dst} = trecip {src}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {vec_type} {vone} = vdupq_n_{suffix}(1.0f);")
        lines.append(f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}        int _col;")
        lines.append(f"{indent}        for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}            {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}            {vec_type} _vr = vdivq_{suffix}({vone}, _v0);")
        lines.append(f"{indent}            vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}        for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}            {dst}[_row][_col] = 1.0f / {src}[_row][_col];")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Transcendental Functions (using scalar fallback)
    # =========================================================================
    
    def gen_texp(self, dst: str, src: str,
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TEXP - elementwise exponential."""
        lines = []
        indent = self.ctx.indent()
        
        # NEON doesn't have native exp, use scalar fallback
        lines.append(f"{indent}// TEXP: {dst} = texp {src}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = expf({src}[_row][_col]);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tlog(self, dst: str, src: str,
                 rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TLOG - elementwise natural log."""
        lines = []
        indent = self.ctx.indent()
        
        lines.append(f"{indent}// TLOG: {dst} = tlog {src}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = logf({src}[_row][_col]);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tsqrt(self, dst: str, src: str,
                  rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TSQRT - elementwise square root."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TSQRT: {dst} = tsqrt {src}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vsqrtq_{suffix}(_v0);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = sqrtf({src}[_row][_col]);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_trsqrt(self, dst: str, src: str,
                   rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TRSQRT - elementwise reciprocal square root."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TRSQRT: {dst} = trsqrt {src}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    int _col;")
        lines.append(f"{indent}    for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}        {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}        {vec_type} _vr = vrsqrteq_{suffix}(_v0);")
        lines.append(f"{indent}        vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        {dst}[_row][_col] = 1.0f / sqrtf({src}[_row][_col]);")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Activation Functions
    # =========================================================================
    
    def gen_trelu(self, dst: str, src: str,
                  rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TRELU - elementwise ReLU."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        vzero = self.ctx.get_unique_var("_vzero")
        
        lines.append(f"{indent}// TRELU: {dst} = trelu {src}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {vec_type} {vzero} = vdupq_n_{suffix}(0.0f);")
        lines.append(f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}        int _col;")
        lines.append(f"{indent}        for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}            {vec_type} _v0 = vld1q_{suffix}(&{src}[_row][_col]);")
        lines.append(f"{indent}            {vec_type} _vr = vmaxq_{suffix}(_v0, {vzero});")
        lines.append(f"{indent}            vst1q_{suffix}(&{dst}[_row][_col], _vr);")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}        for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}            {dst}[_row][_col] = fmaxf({src}[_row][_col], 0.0f);")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Broadcast Operations
    # =========================================================================
    
    def gen_texpands(self, dst: str, scalar: str,
                     rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TEXPANDS - broadcast scalar to tile."""
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        vs = self.ctx.get_unique_var("_vs")
        
        lines.append(f"{indent}// TEXPANDS: {dst} = texpands {scalar}")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    {vec_type} {vs} = vdupq_n_{suffix}({scalar});")
        lines.append(f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}        int _col;")
        lines.append(f"{indent}        for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        lines.append(f"{indent}            vst1q_{suffix}(&{dst}[_row][_col], {vs});")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}        for (; _col < {cols}; _col++) {{")
        lines.append(f"{indent}            {dst}[_row][_col] = {scalar};")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Matrix Operations
    # =========================================================================
    
    def gen_tmatmul(self, dst: str, a: str, b: str,
                    m: int, k: int, n: int, dtype: str = "f32") -> List[str]:
        """
        Generate TMATMUL - matrix multiply.
        C[m,n] = A[m,k] @ B[k,n]
        """
        lines = []
        indent = self.ctx.indent()
        vec_lanes = VECTOR_LANES.get(dtype, 4)
        suffix = self._get_neon_suffix(dtype)
        vec_type = ARM64_VECTOR_TYPE_MAP.get(dtype, "float32x4_t")
        
        lines.append(f"{indent}// TMATMUL: {dst} = tmatmul {a}, {b}")
        lines.append(f"{indent}// Dimensions: [{m}x{k}] @ [{k}x{n}] -> [{m}x{n}]")
        lines.append(f"{indent}for (int _i = 0; _i < {m}; _i++) {{")
        lines.append(f"{indent}    for (int _j = 0; _j < {n}; _j++) {{")
        lines.append(f"{indent}        float _sum = 0.0f;")
        lines.append(f"{indent}        for (int _k = 0; _k < {k}; _k++) {{")
        lines.append(f"{indent}            _sum += {a}[_i][_k] * {b}[_k][_j];")
        lines.append(f"{indent}        }}")
        lines.append(f"{indent}        {dst}[_i][_j] = _sum;")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")
        
        return lines
    
    # =========================================================================
    # Reduction Operations
    # =========================================================================
    
    def gen_trowsum(self, dst: str, src: str,
                    rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TROWSUM - sum reduction across columns."""
        lines = []
        indent = self.ctx.indent()
        
        lines.append(f"{indent}// TROWSUM: {dst} = trowsum {src}")
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}    float _sum = 0.0f;")
        lines.append(f"{indent}    for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"{indent}        _sum += {src}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    {dst}[_row][0] = _sum;")
        lines.append(f"{indent}}}")
        
        return lines
    
    def gen_tcolsum(self, dst: str, src: str,
                    rows: int, cols: int, dtype: str = "f32") -> List[str]:
        """Generate TCOLSUM - sum reduction across rows."""
        lines = []
        indent = self.ctx.indent()
        
        lines.append(f"{indent}// TCOLSUM: {dst} = tcolsum {src}")
        lines.append(f"{indent}for (int _col = 0; _col < {cols}; _col++) {{")
        lines.append(f"{indent}    float _sum = 0.0f;")
        lines.append(f"{indent}    for (int _row = 0; _row < {rows}; _row++) {{")
        lines.append(f"{indent}        _sum += {src}[_row][_col];")
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}    {dst}[0][_col] = _sum;")
        lines.append(f"{indent}}}")
        
        return lines


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_arm64_header() -> str:
    """Generate standard ARM64 NEON header includes."""
    return """// Auto-generated ARM64 NEON code from PTO ISA Compiler
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
"""


def generate_tile_declaration(name: str, rows: int, cols: int, dtype: str = "f32") -> str:
    """Generate C declaration for a tile."""
    c_type = ARM64_TYPE_MAP.get(dtype, "float")
    return f"{c_type} {name}[{rows}][{cols}];"


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    codegen = ARM64CodeGen()
    
    print("ARM64 Code Generator Test")
    print("=" * 60)
    
    # Test TADD
    print("\nTADD Example:")
    lines = codegen.gen_tadd("c", "a", "b", 4, 8, "f32")
    for line in lines:
        print(line)
    
    # Test TMULS
    print("\nTMULS Example:")
    lines = codegen.gen_tmuls("dst", "src", "0.5f", 4, 8, "f32")
    for line in lines:
        print(line)
    
    # Test TEXP
    print("\nTEXP Example:")
    lines = codegen.gen_texp("dst", "src", 4, 8, "f32")
    for line in lines:
        print(line)
