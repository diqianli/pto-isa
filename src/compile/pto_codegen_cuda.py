"""
PTO Compiler - CUDA Code Generation

This module contains CUDA-specific code generation:
- CUDA barrier operation code generation
- CUDA single operation code generation
- CUDA fused loop code generation
- CUDA kernel generation

Dependencies:
- pto_isa_definition: CUDA_TYPE_MAP, cuda_generate_header
- pto_compile_common: Common infrastructure
"""

from typing import Dict, List, Optional, Tuple
import os
import sys

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from isa_definition.pto_isa_definition import CUDA_TYPE_MAP, cuda_generate_header, ElementType

from compile.pto_compile_common import (
    PTOProgram, PTOModule, MockTileInfo, MockInstruction,
    FusedLoop, FusionBarrier, FusableOp, LoopFusionOptimizer,
    TileBufferAnalyzer, convert_program_to_mock_instructions,
)


# =============================================================================
# CUDA Barrier Operation Code Generation
# =============================================================================

def gen_cuda_barrier_op(instr: MockInstruction, rows: int, cols: int, 
                        dtype: str, tile_info: Dict[str, MockTileInfo]) -> List[str]:
    """Generate CUDA code for barrier operations (control flow, scalar, etc.)."""
    lines = []
    c_type = CUDA_TYPE_MAP.get(dtype, "float")
    
    if instr.opcode == "FOR":
        iv = instr.dst
        lb = instr.operands[0]
        ub = instr.operands[1]
        step = instr.operands[2] if len(instr.operands) > 2 else "1"
        lines.append(f"for (int {iv} = {lb}; {iv} < {ub}; {iv} += {step}) {{")
        
    elif instr.opcode == "ENDFOR":
        lines.append("}")
        
    elif instr.opcode == "IF":
        cond = instr.operands[0] if instr.operands else "true"
        lines.append(f"if ({cond}) {{")
        
    elif instr.opcode == "ELSE":
        lines.append("} else {")
        
    elif instr.opcode == "ENDIF":
        lines.append("}")
    
    elif instr.opcode == "SLI":
        dst = instr.dst
        imm = instr.operands[0]
        lines.append(f"int {dst} = {imm};")
        
    elif instr.opcode == "SCMP":
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        cmp_mode = instr.operands[2] if len(instr.operands) > 2 else "eq"
        cmp_ops = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
        cmp_op = cmp_ops.get(cmp_mode, ">")
        lines.append(f"int {dst} = ({src0} {cmp_op} {src1}) ? 1 : 0;")
        
    elif instr.opcode in ("SADD", "SSUB", "SMUL", "SDIV"):
        dst = instr.dst
        src0, src1 = instr.operands[0], instr.operands[1]
        op_map = {"SADD": "+", "SSUB": "-", "SMUL": "*", "SDIV": "/"}
        op = op_map.get(instr.opcode, "+")
        lines.append(f"int {dst} = {src0} {op} {src1};")
        
    elif instr.opcode == "SMOV":
        lines.append(f"int {instr.dst} = {instr.operands[0]};")
    
    elif instr.opcode == "TLOAD":
        dst, src_mem = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        row_offset_expr = f"({row_off}) * {rows}" if row_off != "0" else "0"
        col_offset_expr = col_off if col_off != "0" else "0"
        lines.append(f"// TLOAD: {dst} = load({src_mem}[{row_off}, {col_off}])")
        if row_off == "0" and col_off == "0":
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst}[_row][_col] = {src_mem}[_row * {cols} + _col];")
        else:
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst}[_row][_col] = {src_mem}[({row_offset_expr} + _row) * {cols} + {col_offset_expr} + _col];")
        
    elif instr.opcode == "TSTORE":
        dst_mem, src = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        row_offset_expr = f"({row_off}) * {rows}" if row_off != "0" else "0"
        col_offset_expr = col_off if col_off != "0" else "0"
        lines.append(f"// TSTORE: store({src}) -> {dst_mem}[{row_off}, {col_off}]")
        if row_off == "0" and col_off == "0":
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst_mem}[_row * {cols} + _col] = {src}[_row][_col];")
        else:
            lines.append(f"if (_row < {rows} && _col < {cols}) {dst_mem}[({row_offset_expr} + _row) * {cols} + {col_offset_expr} + _col] = {src}[_row][_col];")
        
    elif instr.opcode == "TROWSUM":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_cols = src_info.cols if src_info else cols
        lines.append(f"// TROWSUM: {dst} = rowsum({src})")
        lines.append(f"if (_col == 0 && _row < {rows}) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _c = 0; _c < {src_cols}; _c++) _sum += {src}[_row][_c];")
        lines.append(f"    {dst}[_row][0] = _sum;}}")
    
    elif instr.opcode == "TROWMAX":
        dst, src = instr.dst, instr.operands[0]
        src_info = tile_info.get(src)
        src_cols = src_info.cols if src_info else cols
        lines.append(f"// TROWMAX: {dst} = rowmax({src})")
        lines.append(f"if (_col == 0 && _row < {rows}) {{")
        lines.append(f"    {c_type} _max = {src}[_row][0];")
        lines.append(f"    for (int _c = 1; _c < {src_cols}; _c++) if ({src}[_row][_c] > _max) _max = {src}[_row][_c];")
        lines.append(f"    {dst}[_row][0] = _max;}}")
        
    elif instr.opcode == "TMATMUL":
        dst, a, b = instr.dst, instr.operands[0], instr.operands[1]
        a_info = tile_info.get(a)
        k = a_info.cols if a_info else 8
        lines.append(f"// TMATMUL: {dst} = {a} @ {b}")
        lines.append(f"if (_row < {rows} && _col < {cols}) {{")
        lines.append(f"    {c_type} _sum = 0.0f;")
        lines.append(f"    for (int _k = 0; _k < {k}; _k++) _sum += {a}[_row][_k] * {b}[_k][_col];")
        lines.append(f"    {dst}[_row][_col] = _sum;}}")
    
    elif instr.opcode == "CALL":
        callee = instr.dst
        args = instr.operands
        if args:
            args_str = ", ".join(str(arg) for arg in args)
            lines.append(f"{callee}({args_str});")
        else:
            lines.append(f"{callee}();")
    
    elif instr.opcode == "RETURN":
        lines.append("return;")
        
    else:
        lines.append(f"// {instr.opcode}: Not implemented")
    
    return lines


# =============================================================================
# CUDA Single Operation Code Generation
# =============================================================================

def gen_cuda_single_op(instr: MockInstruction, tile_info: Dict[str, MockTileInfo]) -> str:
    """Generate a single CUDA operation."""
    op, dst = instr.opcode, f"{instr.dst}[_row][_col]"
    src0 = src1 = ""
    
    if len(instr.operands) >= 1:
        src0 = f"{instr.operands[0]}[_row][_col]"
    if len(instr.operands) >= 2:
        src1 = instr.operands[1]
        if not src1.endswith("f") and not src1.replace(".", "").replace("-", "").isdigit():
            src1 = f"{src1}[_row][_col]"
    
    dst_info = tile_info.get(instr.dst)
    rows = dst_info.rows if dst_info else 8
    cols = dst_info.cols if dst_info else 8
    tile_size = rows * cols
    
    def compute_mem_idx(operands, cols, tile_size):
        row_off = operands[1] if len(operands) > 1 else "0"
        col_off = operands[2] if len(operands) > 2 else "0"
        if row_off == "0" and col_off == "0":
            return f"_row * {cols} + _col"
        else:
            row_offset = f"({row_off}) * {tile_size}" if row_off != "0" else ""
            col_offset = col_off if col_off != "0" else ""
            base_idx = f"_row * {cols} + _col"
            if row_offset and col_offset:
                return f"{row_offset} + {col_offset} + {base_idx}"
            elif row_offset:
                return f"{row_offset} + {base_idx}"
            elif col_offset:
                return f"{col_offset} + {base_idx}"
            else:
                return base_idx
    
    # Binary operations
    if op == "TADD": return f"{dst} = {src0} + {src1};"
    elif op == "TSUB": return f"{dst} = {src0} - {src1};"
    elif op == "TMUL": return f"{dst} = {src0} * {src1};"
    elif op == "TDIV": return f"{dst} = {src0} / {src1};"
    elif op == "TMAX": return f"{dst} = fmaxf({src0}, {src1});"
    elif op == "TMIN": return f"{dst} = fminf({src0}, {src1});"
    
    # Unary operations
    elif op == "TABS": return f"{dst} = fabsf({src0});"
    elif op == "TNEG": return f"{dst} = -{src0};"
    elif op == "TRECIP": return f"{dst} = 1.0f / {src0};"
    elif op == "TEXP": return f"{dst} = __expf({src0});"
    elif op == "TLOG": return f"{dst} = __logf({src0});"
    elif op == "TSQRT": return f"{dst} = __fsqrt_rn({src0});"
    elif op == "TRSQRT": return f"{dst} = __frsqrt_rn({src0});"
    elif op == "TRELU": return f"{dst} = fmaxf({src0}, 0.0f);"
    
    # Broadcast binary operations
    elif op == "TROWEXPANDSUB":
        return f"{dst} = {instr.operands[0]}[_row][_col] - {instr.operands[1]}[_row][0];"
    elif op == "TROWEXPANDDIV":
        return f"{dst} = {instr.operands[0]}[_row][_col] / {instr.operands[1]}[_row][0];"
    elif op == "TROWEXPANDMUL":
        return f"{dst} = {instr.operands[0]}[_row][_col] * {instr.operands[1]}[_row][0];"
    
    # Scalar operations
    elif op == "TADDS": return f"{dst} = {src0} + {src1};"
    elif op == "TSUBS": return f"{dst} = {src0} - {src1};"
    elif op == "TMULS": return f"{dst} = {src0} * {src1};"
    elif op == "TDIVS": return f"{dst} = {src0} / {src1};"
    elif op == "TEXPANDS": return f"{dst} = {instr.operands[0]};"
    
    # Memory operations
    elif op == "TLOAD": 
        memref = instr.operands[0]
        mem_idx = compute_mem_idx(instr.operands, cols, tile_size)
        return f"{dst} = {memref}[{mem_idx}];"
    elif op == "TSTORE": 
        src_info = tile_info.get(instr.operands[0])
        src_cols = src_info.cols if src_info else cols
        src_rows = src_info.rows if src_info else rows
        src_tile_size = src_rows * src_cols
        mem_idx = compute_mem_idx(instr.operands, src_cols, src_tile_size)
        return f"{instr.dst}[{mem_idx}] = {instr.operands[0]}[_row][_col];"
    
    return f"// Unknown op: {op}"


# =============================================================================
# CUDA Fused Loop Code Generation
# =============================================================================

class CUDAFusedCodeGenerator:
    """Generate CUDA code for fused loops."""
    
    def __init__(self):
        self.dtype_map = CUDA_TYPE_MAP
    
    def generate_fused_loop(self, fused_loop: FusedLoop) -> List[str]:
        """Generate CUDA kernel code for a fused loop."""
        lines = []
        rows = fused_loop.tile_shape.rows
        cols = fused_loop.tile_shape.cols
        c_type = self.dtype_map.get(fused_loop.dtype, "float")
        
        lines.append(f"// Fused loop: {len(fused_loop.operations)} operations")
        lines.append(f"if (_row < {rows} && _col < {cols}) {{")
        
        for op in fused_loop.operations:
            op_code = self._generate_single_op(op, c_type)
            lines.append(f"    {op_code}")
        
        lines.append("}")
        
        return lines
    
    def _generate_single_op(self, op: FusableOp, c_type: str) -> str:
        """Generate CUDA code for a single fusable operation."""
        dst = f"{op.dst}[_row][_col]"
        src0 = f"{op.operands[0]}[_row][_col]" if op.operands else ""
        src1 = ""
        
        if len(op.operands) >= 2:
            src1_val = op.operands[1]
            if isinstance(src1_val, str) and not src1_val.replace(".", "").replace("-", "").isdigit():
                src1 = f"{src1_val}[_row][_col]"
            else:
                src1 = str(src1_val)
        
        # Binary operations
        if op.opcode == "TADD": return f"{dst} = {src0} + {src1};"
        elif op.opcode == "TSUB": return f"{dst} = {src0} - {src1};"
        elif op.opcode == "TMUL": return f"{dst} = {src0} * {src1};"
        elif op.opcode == "TDIV": return f"{dst} = {src0} / {src1};"
        elif op.opcode == "TMAX": return f"{dst} = fmaxf({src0}, {src1});"
        elif op.opcode == "TMIN": return f"{dst} = fminf({src0}, {src1});"
        
        # Unary operations
        elif op.opcode == "TABS": return f"{dst} = fabsf({src0});"
        elif op.opcode == "TNEG": return f"{dst} = -{src0};"
        elif op.opcode == "TRECIP": return f"{dst} = 1.0f / {src0};"
        elif op.opcode == "TEXP": return f"{dst} = __expf({src0});"
        elif op.opcode == "TLOG": return f"{dst} = __logf({src0});"
        elif op.opcode == "TSQRT": return f"{dst} = __fsqrt_rn({src0});"
        elif op.opcode == "TRSQRT": return f"{dst} = __frsqrt_rn({src0});"
        elif op.opcode == "TRELU": return f"{dst} = fmaxf({src0}, 0.0f);"
        elif op.opcode == "TSIGMOID": return f"{dst} = 1.0f / (1.0f + __expf(-{src0}));"
        elif op.opcode == "TTANH": return f"{dst} = tanhf({src0});"
        
        # Scalar operations
        elif op.opcode == "TADDS": return f"{dst} = {src0} + {src1};"
        elif op.opcode == "TSUBS": return f"{dst} = {src0} - {src1};"
        elif op.opcode == "TMULS": return f"{dst} = {src0} * {src1};"
        elif op.opcode == "TDIVS": return f"{dst} = {src0} / {src1};"
        elif op.opcode == "TEXPANDS": return f"{dst} = {op.operands[0]};"
        
        return f"// Unknown op: {op.opcode}"


# =============================================================================
# CUDA Code Generator
# =============================================================================

class CUDACodeGenerator:
    """
    Generates CUDA code from PTO programs.
    """
    
    def __init__(self, enable_fusion: bool = True, analyze_buffers: bool = True,
                 module: Optional['PTOModule'] = None):
        self.enable_fusion = enable_fusion
        self.analyze_buffers = analyze_buffers
        self.module = module
    
    def generate(self, program: PTOProgram) -> str:
        """Generate CUDA code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        is_in_core = getattr(program, 'is_in_core', True)
        
        lines = [
            f"// PTO Program: {program.name}",
            f"// Target: NVIDIA CUDA",
        ]
        
        # Add buffer analysis for InCore functions
        if is_in_core and self.analyze_buffers:
            analyzer = TileBufferAnalyzer(program)
            analyzer.analyze()
            report = analyzer.generate_report()
            lines.append(report)
        
        lines.append(cuda_generate_header())
        
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in program.memref_declarations.items():
            c_type = CUDA_TYPE_MAP.get(memref_type.element_type.value, "float")
            memref_params.append(f"{c_type}* {name}")
        
        # Find scalars initialized by SLI
        sli_initialized_scalars = set()
        for instr in mock_instructions:
            if instr.opcode == "SLI":
                sli_initialized_scalars.add(instr.dst)
        
        # Declare scalar variables as function parameters
        scalar_params = []
        for name, scalar_type in program.scalar_declarations.items():
            if scalar_type in (ElementType.U1, ElementType.INDEX):
                continue
            if name in sli_initialized_scalars:
                continue
            c_type = CUDA_TYPE_MAP.get(scalar_type.value, "int")
            scalar_params.append(f"{c_type} {name}")
        
        all_params = memref_params + scalar_params
        
        # Generate kernel signature
        if all_params:
            func_params = ", ".join(all_params)
            lines.append(f"__global__ void {program.name}({func_params}) {{")
        else:
            lines.append(f"__global__ void {program.name}(void) {{")
        
        # Thread indices
        lines.append("    int _row = blockIdx.y * blockDim.y + threadIdx.y;")
        lines.append("    int _col = blockIdx.x * blockDim.x + threadIdx.x;")
        lines.append("")
        
        # Declare tiles as shared memory
        for name, info in tile_info.items():
            c_type = CUDA_TYPE_MAP.get(info.dtype, "float")
            lines.append(f"    __shared__ {c_type} {name}[{info.rows}][{info.cols}];")
        lines.append("")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            fused_codegen = CUDAFusedCodeGenerator()
            indent_level = 1
            
            for item in fused_result:
                indent = "    " * indent_level
                
                if isinstance(item, FusedLoop):
                    fused_lines = fused_codegen.generate_fused_loop(item)
                    for fused_line in fused_lines:
                        lines.append(f"{indent}{fused_line}" if fused_line else "")
                    lines.append("")
                elif isinstance(item, FusionBarrier):
                    instr = item.raw_instr
                    info = tile_info.get(instr.dst) if instr.dst else None
                    rows = info.rows if info else 8
                    cols = info.cols if info else 8
                    dtype = info.dtype if info else "f32"
                    
                    # Handle indentation for control flow
                    if instr.opcode in ("ENDFOR", "ENDIF"):
                        indent_level = max(1, indent_level - 1)
                        indent = "    " * indent_level
                    elif instr.opcode == "ELSE":
                        indent = "    " * max(1, indent_level - 1)
                    
                    barrier_lines = gen_cuda_barrier_op(instr, rows, cols, dtype, tile_info)
                    for barrier_line in barrier_lines:
                        lines.append(f"{indent}{barrier_line}" if barrier_line else "")
                    
                    if instr.opcode in ("FOR", "IF", "ELSE"):
                        indent_level += 1
                    
                    lines.append("")
        
        lines.append("}")
        
        return "\n".join(lines)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'gen_cuda_barrier_op',
    'gen_cuda_single_op',
    'CUDAFusedCodeGenerator',
    'CUDACodeGenerator',
]
