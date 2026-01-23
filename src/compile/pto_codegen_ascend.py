"""
PTO Compiler - Ascend Code Generation

This module contains Ascend (Huawei NPU)-specific code generation:
- Ascend C barrier operation code generation
- Ascend C single operation code generation  
- Ascend fused loop code generation
- Support for A2/A3 and A5 architectures

Dependencies:
- pto_isa_definition: ASCEND_TYPE_MAP, ascend_generate_header
- pto_compile_common: Common infrastructure
"""

from typing import Dict, List, Optional
import os
import sys

# Add parent directories to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from isa_definition.pto_isa_definition import ASCEND_TYPE_MAP, ascend_generate_header, ElementType

from compile.pto_compile_common import (
    PTOProgram, PTOModule, MockTileInfo, MockInstruction,
    FusedLoop, FusionBarrier, FusableOp, LoopFusionOptimizer,
    TileBufferAnalyzer, convert_program_to_mock_instructions,
)


# =============================================================================
# Ascend Barrier Operation Code Generation
# =============================================================================

def gen_ascend_barrier_op(instr: MockInstruction, rows: int, cols: int, 
                          dtype: str, tile_info: Dict[str, MockTileInfo]) -> List[str]:
    """Generate Ascend C code for barrier operations (control flow, scalar, etc.)."""
    lines = []
    
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
        lines.append(f"int {instr.dst} = {instr.operands[0]};")
        
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
        tile_size = rows * cols
        lines.append(f"// TLOAD: {dst} = load({src_mem}[{row_off}, {col_off}])")
        lines.append(f"DataCopy({dst}, {src_mem}[({row_off}) * {tile_size}], {tile_size});")
        
    elif instr.opcode == "TSTORE":
        dst_mem, src = instr.dst, instr.operands[0]
        row_off = instr.operands[1] if len(instr.operands) > 1 else "0"
        col_off = instr.operands[2] if len(instr.operands) > 2 else "0"
        tile_size = rows * cols
        lines.append(f"// TSTORE: store({src}) -> {dst_mem}[{row_off}, {col_off}]")
        lines.append(f"DataCopy({dst_mem}[({row_off}) * {tile_size}], {src}, {tile_size});")
        
    elif instr.opcode == "TROWSUM":
        tile_size = rows * cols
        lines.append(f"// TROWSUM: reduction operation")
        lines.append(f"ReduceSum({instr.dst}, {instr.operands[0]}, {tile_size});")
    
    elif instr.opcode == "TROWMAX":
        tile_size = rows * cols
        lines.append(f"// TROWMAX: reduction max operation")
        lines.append(f"ReduceMax({instr.dst}, {instr.operands[0]}, {tile_size});")
        
    elif instr.opcode == "TMATMUL":
        lines.append(f"// TMATMUL: {instr.dst} = {instr.operands[0]} @ {instr.operands[1]}")
        lines.append(f"Matmul({instr.dst}, {instr.operands[0]}, {instr.operands[1]}, {rows}, {cols});")
    
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
# Ascend Single Operation Code Generation
# =============================================================================

def gen_ascend_single_op(instr: MockInstruction, tile_info: Dict[str, MockTileInfo]) -> str:
    """Generate a single Ascend C vector operation."""
    op, dst = instr.opcode, instr.dst
    src0 = instr.operands[0] if len(instr.operands) >= 1 else ""
    src1 = instr.operands[1] if len(instr.operands) >= 2 else ""
    
    ops_map = {
        # Binary operations
        "TADD": f"Add({dst}, {src0}, {src1}, 64);",
        "TSUB": f"Sub({dst}, {src0}, {src1}, 64);",
        "TMUL": f"Mul({dst}, {src0}, {src1}, 64);",
        "TDIV": f"Div({dst}, {src0}, {src1}, 64);",
        "TMAX": f"Max({dst}, {src0}, {src1}, 64);",
        "TMIN": f"Min({dst}, {src0}, {src1}, 64);",
        
        # Unary operations
        "TABS": f"Abs({dst}, {src0}, 64);",
        "TNEG": f"Neg({dst}, {src0}, 64);",
        "TRECIP": f"Reciprocal({dst}, {src0}, 64);",
        "TEXP": f"Exp({dst}, {src0}, 64);",
        "TLOG": f"Ln({dst}, {src0}, 64);",
        "TSQRT": f"Sqrt({dst}, {src0}, 64);",
        "TRSQRT": f"Rsqrt({dst}, {src0}, 64);",
        "TRELU": f"Relu({dst}, {src0}, 64);",
        
        # Scalar operations
        "TADDS": f"Adds({dst}, {src0}, {src1}, 64);",
        "TSUBS": f"Subs({dst}, {src0}, {src1}, 64);",
        "TMULS": f"Muls({dst}, {src0}, {src1}, 64);",
        "TDIVS": f"Divs({dst}, {src0}, {src1}, 64);",
        "TEXPANDS": f"Duplicate({dst}, {src0}, 64);",
        
        # Row-wise broadcast operations
        "TROWEXPANDSUB": f"BroadcastSub({dst}, {src0}, {src1}, 64, 8);  // row-wise broadcast subtract",
        "TROWEXPANDDIV": f"BroadcastDiv({dst}, {src0}, {src1}, 64, 8);  // row-wise broadcast divide",
        "TROWEXPANDMUL": f"BroadcastMul({dst}, {src0}, {src1}, 64, 8);  // row-wise broadcast multiply",
    }
    return ops_map.get(op, f"// {op}: Operation")


# =============================================================================
# Ascend Fused Loop Code Generation
# =============================================================================

class AscendFusedCodeGenerator:
    """Generate Ascend C code for fused loops."""
    
    def __init__(self):
        self.dtype_map = ASCEND_TYPE_MAP
    
    def generate_fused_loop(self, fused_loop: FusedLoop) -> List[str]:
        """Generate Ascend C code for a fused loop."""
        lines = []
        rows = fused_loop.tile_shape.rows
        cols = fused_loop.tile_shape.cols
        tile_size = rows * cols
        
        lines.append(f"// Fused vector operations: {len(fused_loop.operations)} operations")
        lines.append(f"// Tile size: {rows}x{cols} = {tile_size} elements")
        
        for op in fused_loop.operations:
            op_code = self._generate_single_op(op, tile_size)
            lines.append(op_code)
        
        return lines
    
    def _generate_single_op(self, op: FusableOp, tile_size: int) -> str:
        """Generate Ascend C code for a single fusable operation."""
        dst = op.dst
        src0 = op.operands[0] if op.operands else ""
        src1 = op.operands[1] if len(op.operands) >= 2 else ""
        
        # Binary operations
        if op.opcode == "TADD": return f"Add({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TSUB": return f"Sub({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TMUL": return f"Mul({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TDIV": return f"Div({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TMAX": return f"Max({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TMIN": return f"Min({dst}, {src0}, {src1}, {tile_size});"
        
        # Unary operations
        elif op.opcode == "TABS": return f"Abs({dst}, {src0}, {tile_size});"
        elif op.opcode == "TNEG": return f"Neg({dst}, {src0}, {tile_size});"
        elif op.opcode == "TRECIP": return f"Reciprocal({dst}, {src0}, {tile_size});"
        elif op.opcode == "TEXP": return f"Exp({dst}, {src0}, {tile_size});"
        elif op.opcode == "TLOG": return f"Ln({dst}, {src0}, {tile_size});"
        elif op.opcode == "TSQRT": return f"Sqrt({dst}, {src0}, {tile_size});"
        elif op.opcode == "TRSQRT": return f"Rsqrt({dst}, {src0}, {tile_size});"
        elif op.opcode == "TRELU": return f"Relu({dst}, {src0}, {tile_size});"
        elif op.opcode == "TSIGMOID": return f"Sigmoid({dst}, {src0}, {tile_size});"
        elif op.opcode == "TTANH": return f"Tanh({dst}, {src0}, {tile_size});"
        elif op.opcode == "TGELU": return f"Gelu({dst}, {src0}, {tile_size});"
        elif op.opcode == "TSILU": return f"Swish({dst}, {src0}, {tile_size});"
        
        # Scalar operations
        elif op.opcode == "TADDS": return f"Adds({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TSUBS": return f"Subs({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TMULS": return f"Muls({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TDIVS": return f"Divs({dst}, {src0}, {src1}, {tile_size});"
        elif op.opcode == "TEXPANDS": return f"Duplicate({dst}, {src0}, {tile_size});"
        
        return f"// Unknown op: {op.opcode}"


# =============================================================================
# Ascend Code Generator
# =============================================================================

class AscendCodeGenerator:
    """
    Generates Ascend C code from PTO programs.
    Supports A2/A3 and A5 architectures.
    """
    
    def __init__(self, enable_fusion: bool = True, analyze_buffers: bool = True,
                 module: Optional['PTOModule'] = None, target: str = "a2a3"):
        self.enable_fusion = enable_fusion
        self.analyze_buffers = analyze_buffers
        self.module = module
        self.target = target  # "a2a3" or "a5"
    
    def generate(self, program: PTOProgram) -> str:
        """Generate Ascend C code from a PTO program."""
        tile_info, mock_instructions = convert_program_to_mock_instructions(program)
        
        is_in_core = getattr(program, 'is_in_core', True)
        
        target_name = "Ascend A2/A3" if self.target == "a2a3" else "Ascend A5"
        lines = [
            f"// PTO Program: {program.name}",
            f"// Target: {target_name}",
        ]
        
        # Add buffer analysis for InCore functions
        if is_in_core and self.analyze_buffers:
            analyzer = TileBufferAnalyzer(program)
            analyzer.analyze()
            report = analyzer.generate_report()
            lines.append(report)
        
        lines.append(ascend_generate_header())
        
        # Collect memory references for function parameters
        memref_params = []
        for name, memref_type in program.memref_declarations.items():
            c_type = ASCEND_TYPE_MAP.get(memref_type.element_type.value, "float")
            memref_params.append(f"__gm__ {c_type}* {name}")
        
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
            c_type = ASCEND_TYPE_MAP.get(scalar_type.value, "int")
            scalar_params.append(f"{c_type} {name}")
        
        all_params = memref_params + scalar_params
        
        # Generate function signature
        if all_params:
            func_params = ", ".join(all_params)
            lines.append(f"__aicore__ void {program.name}({func_params}) {{")
        else:
            lines.append(f"__aicore__ void {program.name}(void) {{")
        
        # Declare tiles as local memory
        for name, info in tile_info.items():
            c_type = ASCEND_TYPE_MAP.get(info.dtype, "float")
            lines.append(f"    __ub__ {c_type} {name}[{info.rows * info.cols}];")
        lines.append("")
        
        if self.enable_fusion:
            optimizer = LoopFusionOptimizer(tile_info)
            fused_result = optimizer.optimize(mock_instructions)
            lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved\n")
            
            fused_codegen = AscendFusedCodeGenerator()
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
                    
                    barrier_lines = gen_ascend_barrier_op(instr, rows, cols, dtype, tile_info)
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
    'gen_ascend_barrier_op',
    'gen_ascend_single_op',
    'AscendFusedCodeGenerator',
    'AscendCodeGenerator',
]
