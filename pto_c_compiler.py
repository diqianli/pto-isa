"""
PTO to ARM64 C Compiler

This compiler takes a C file containing PTO assembly-like syntax and generates
a C file with ARM64 NEON intrinsics.

Features:
- Loop fusion optimization: combines consecutive elementwise operations
- NEON vectorization: generates ARM64 NEON intrinsics for SIMD execution
- Architecture-agnostic design: uses codegen_arm64() method on each instruction

Input: .c file with PTO pseudo-instructions in comments/macros
Output: .c file with ARM64 NEON vectorized code

Usage:
    python pto_c_compiler.py input.c output.c
    python pto_c_compiler.py input.c output.c --no-fusion  # Disable fusion
"""

import re
import sys
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Import instruction classes and ARM64 codegen infrastructure from pto_isa_definition
from pto_isa_definition import (
    # Codegen infrastructure
    ARM64CodeGenContext, ARM64_TYPE_MAP, arm64_generate_header,
    # Types
    TileType, TileShape, TileOperand, ScalarOperand, ImmediateOperand,
    MemRefOperand, MemRefType, MemorySpace, ElementType,
    # Instructions
    TADD, TSUB, TMUL, TDIV, TMAX, TMIN,
    TABS, TNEG, TEXP, TLOG, TSQRT, TRSQRT, TRECIP, TRELU,
    TADDS, TMULS, TDIVS,
    TEXPANDS, TMATMUL, TROWSUM, TCOLSUM, TLOAD, TSTORE,
)

# Backward compatibility: import ARM64CodeGen for loop fusion
from arm64_codegen import ARM64CodeGen, generate_arm64_header
from loop_fusion import (
    LoopFusionOptimizer, FusedLoop, FusionBarrier, FusedCodeGenerator,
    is_fusion_barrier, get_category, OpCategory
)


# =============================================================================
# PTO Instruction Parser
# =============================================================================

@dataclass
class TileInfo:
    """Information about a declared tile."""
    name: str
    rows: int
    cols: int
    dtype: str


@dataclass
class ParsedInstruction:
    """A parsed PTO instruction."""
    opcode: str
    dst: Optional[str]
    operands: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""


class PTOParser:
    """
    Parser for PTO instructions in C code.
    
    Recognizes patterns like:
    - PTO_TILE(name, rows, cols, dtype)  // Tile declaration
    - PTO_TADD(dst, src0, src1)          // Instructions
    - PTO_FOR(iv, lb, ub, step) { ... } PTO_ENDFOR
    """
    
    # Regex patterns for PTO macros
    TILE_DECL_PATTERN = re.compile(
        r'PTO_TILE\s*\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)'
    )
    
    SCALAR_DECL_PATTERN = re.compile(
        r'PTO_SCALAR\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
    )
    
    # Binary operations: PTO_OP(dst, src0, src1)
    BINARY_OP_PATTERN = re.compile(
        r'PTO_(TADD|TSUB|TMUL|TDIV|TMAX|TMIN|TAND|TOR|TXOR)\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)'
    )
    
    # Scalar operations: PTO_OP(dst, src, scalar)
    SCALAR_OP_PATTERN = re.compile(
        r'PTO_(TADDS|TSUBS|TMULS|TDIVS|TMAXS|TMINS)\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*([^)]+)\s*\)'
    )
    
    # Unary operations: PTO_OP(dst, src)
    UNARY_OP_PATTERN = re.compile(
        r'PTO_(TABS|TNEG|TRECIP|TEXP|TLOG|TSQRT|TRSQRT|TRELU)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
    )
    
    # Broadcast: PTO_TEXPANDS(dst, scalar)
    BROADCAST_PATTERN = re.compile(
        r'PTO_TEXPANDS\s*\(\s*(\w+)\s*,\s*([^)]+)\s*\)'
    )
    
    # Load/Store: PTO_TLOAD(dst, src, row, col) / PTO_TSTORE(src, dst, row, col)
    LOAD_PATTERN = re.compile(
        r'PTO_TLOAD\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
    )
    
    STORE_PATTERN = re.compile(
        r'PTO_TSTORE\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
    )
    
    # Matrix multiply: PTO_TMATMUL(dst, a, b)
    MATMUL_PATTERN = re.compile(
        r'PTO_TMATMUL\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)'
    )
    
    # Reduction: PTO_TROWSUM(dst, src) / PTO_TCOLSUM(dst, src)
    REDUCTION_PATTERN = re.compile(
        r'PTO_(TROWSUM|TCOLSUM)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
    )
    
    # Control flow
    FOR_PATTERN = re.compile(
        r'PTO_FOR\s*\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
    )
    
    ENDFOR_PATTERN = re.compile(r'PTO_ENDFOR')
    
    def __init__(self):
        self.tiles: Dict[str, TileInfo] = {}
        self.scalars: Dict[str, str] = {}
    
    def parse_line(self, line: str) -> Optional[ParsedInstruction]:
        """Parse a single line for PTO instructions."""
        line = line.strip()
        
        # Skip empty lines and pure comments
        if not line or line.startswith('//'):
            return None
        
        # Check for tile declaration
        match = self.TILE_DECL_PATTERN.search(line)
        if match:
            name, rows, cols, dtype = match.groups()
            self.tiles[name] = TileInfo(name, int(rows), int(cols), dtype)
            return ParsedInstruction(
                opcode="TILE_DECL",
                dst=name,
                operands=[rows, cols, dtype],
                raw_line=line
            )
        
        # Check for scalar declaration
        match = self.SCALAR_DECL_PATTERN.search(line)
        if match:
            name, dtype = match.groups()
            self.scalars[name] = dtype
            return ParsedInstruction(
                opcode="SCALAR_DECL",
                dst=name,
                operands=[dtype],
                raw_line=line
            )
        
        # Check for binary operations
        match = self.BINARY_OP_PATTERN.search(line)
        if match:
            op, dst, src0, src1 = match.groups()
            return ParsedInstruction(
                opcode=op,
                dst=dst,
                operands=[src0, src1],
                raw_line=line
            )
        
        # Check for scalar operations
        match = self.SCALAR_OP_PATTERN.search(line)
        if match:
            op, dst, src, scalar = match.groups()
            return ParsedInstruction(
                opcode=op,
                dst=dst,
                operands=[src, scalar.strip()],
                raw_line=line
            )
        
        # Check for unary operations
        match = self.UNARY_OP_PATTERN.search(line)
        if match:
            op, dst, src = match.groups()
            return ParsedInstruction(
                opcode=op,
                dst=dst,
                operands=[src],
                raw_line=line
            )
        
        # Check for broadcast
        match = self.BROADCAST_PATTERN.search(line)
        if match:
            dst, scalar = match.groups()
            return ParsedInstruction(
                opcode="TEXPANDS",
                dst=dst,
                operands=[scalar.strip()],
                raw_line=line
            )
        
        # Check for load
        match = self.LOAD_PATTERN.search(line)
        if match:
            dst, src, row, col = match.groups()
            return ParsedInstruction(
                opcode="TLOAD",
                dst=dst,
                operands=[src, row, col],
                raw_line=line
            )
        
        # Check for store
        match = self.STORE_PATTERN.search(line)
        if match:
            src, dst, row, col = match.groups()
            return ParsedInstruction(
                opcode="TSTORE",
                dst=dst,
                operands=[src, row, col],
                raw_line=line
            )
        
        # Check for matmul
        match = self.MATMUL_PATTERN.search(line)
        if match:
            dst, a, b = match.groups()
            return ParsedInstruction(
                opcode="TMATMUL",
                dst=dst,
                operands=[a, b],
                raw_line=line
            )
        
        # Check for reduction
        match = self.REDUCTION_PATTERN.search(line)
        if match:
            op, dst, src = match.groups()
            return ParsedInstruction(
                opcode=op,
                dst=dst,
                operands=[src],
                raw_line=line
            )
        
        # Check for FOR loop
        match = self.FOR_PATTERN.search(line)
        if match:
            iv, lb, ub, step = match.groups()
            return ParsedInstruction(
                opcode="FOR",
                dst=None,
                operands=[iv, lb, ub, step],
                raw_line=line
            )
        
        # Check for ENDFOR
        match = self.ENDFOR_PATTERN.search(line)
        if match:
            return ParsedInstruction(
                opcode="ENDFOR",
                dst=None,
                operands=[],
                raw_line=line
            )
        
        # Not a PTO instruction - return as passthrough
        return ParsedInstruction(
            opcode="PASSTHROUGH",
            dst=None,
            operands=[],
            raw_line=line
        )
    
    def get_tile_info(self, name: str) -> Optional[TileInfo]:
        """Get tile information by name."""
        return self.tiles.get(name)


# =============================================================================
# PTO to ARM64 Compiler
# =============================================================================

class PTOToARM64Compiler:
    """
    Compiles PTO C code to ARM64 NEON C code.
    
    Features:
    - Loop fusion: combines consecutive elementwise ops into single loops
    - NEON vectorization: generates ARM64 NEON intrinsics
    - Architecture-agnostic: calls codegen_arm64() on instruction objects
    """
    
    def __init__(self, enable_fusion: bool = True, use_instruction_codegen: bool = True):
        """
        Initialize the compiler.
        
        Args:
            enable_fusion: Whether to enable loop fusion optimization
            use_instruction_codegen: Use codegen_arm64() on instruction objects (True)
                                     or use legacy ARM64CodeGen class (False)
        """
        self.parser = PTOParser()
        self.codegen = ARM64CodeGen()  # Legacy codegen for backward compatibility
        self.fused_codegen = FusedCodeGenerator()
        self.arm64_ctx = ARM64CodeGenContext()  # Context for instruction codegen
        self.output_lines: List[str] = []
        self.indent_level = 0
        self.enable_fusion = enable_fusion
        self.use_instruction_codegen = use_instruction_codegen
    
    def compile(self, input_code: str) -> str:
        """
        Compile PTO C code to ARM64 C code.
        
        Args:
            input_code: The input C file content with PTO macros
            
        Returns:
            Generated ARM64 C code
        """
        self.output_lines = []
        self.indent_level = 0
        self.parser = PTOParser()
        self.codegen = ARM64CodeGen()
        self.fused_codegen = FusedCodeGenerator()
        
        # Add header
        self.output_lines.append(generate_arm64_header())
        
        # Process line by line
        lines = input_code.split('\n')
        in_pto_function = False
        function_instructions = []
        
        for line in lines:
            # Check if we're entering a function with PTO code
            if 'PTO_FUNCTION_START' in line:
                in_pto_function = True
                function_instructions = []
                # Output the line without the marker
                remaining = line.replace('PTO_FUNCTION_START', '').strip()
                if remaining:
                    self.output_lines.append(remaining)
                continue
            
            if 'PTO_FUNCTION_END' in line:
                in_pto_function = False
                # Output the line without the marker
                remaining = line.replace('PTO_FUNCTION_END', '').strip()
                if remaining:
                    self.output_lines.append(remaining)
                continue
            
            # Parse the line
            instr = self.parser.parse_line(line)
            
            if instr is None:
                # Empty or comment line - always output
                self.output_lines.append(line)
                continue
            
            if instr.opcode == "PASSTHROUGH":
                stripped = instr.raw_line.strip()
                
                # Check if this is the function closing brace
                if in_pto_function and self.enable_fusion and stripped == '}':
                    # Emit fused instructions BEFORE the closing brace
                    if function_instructions:
                        self._process_with_fusion(function_instructions)
                        function_instructions = []
                    self.output_lines.append(instr.raw_line)
                else:
                    # Output as-is (function signature, other C code)
                    self.output_lines.append(instr.raw_line)
                continue
            
            if in_pto_function and self.enable_fusion:
                # Collect instructions for fusion
                function_instructions.append(instr)
            else:
                # Generate ARM64 code immediately (no fusion)
                generated = self._generate_instruction(instr)
                self.output_lines.extend(generated)
        
        return '\n'.join(self.output_lines)
    
    def _process_with_fusion(self, instructions: List[ParsedInstruction]):
        """
        Process instructions with loop fusion optimization.
        
        Args:
            instructions: List of parsed instructions to optimize
        """
        # Create fusion optimizer with tile information
        optimizer = LoopFusionOptimizer(self.parser.tiles)
        
        # Apply fusion optimization
        fused_result = optimizer.optimize(instructions)
        
        # Print fusion statistics
        if optimizer.stats["fusion_savings"] > 0:
            self.output_lines.append(f"    // Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved")
        
        # Generate code for each fused/unfused group
        for item in fused_result:
            if isinstance(item, FusedLoop):
                # Generate fused loop code
                self.fused_codegen.indent_level = self.indent_level
                lines = self.fused_codegen.generate_fused_loop(item)
                self.output_lines.extend(lines)
            elif isinstance(item, FusionBarrier):
                # Generate code for barrier instruction
                generated = self._generate_instruction(item.raw_instr)
                self.output_lines.extend(generated)
            elif hasattr(item, 'opcode'):
                # Regular instruction (declarations, etc.)
                generated = self._generate_instruction(item)
                self.output_lines.extend(generated)
    
    def _get_indent(self) -> str:
        """Get current indentation."""
        return "    " * self.indent_level
    
    def _create_tile_operand(self, name: str, rows: int, cols: int, dtype: str) -> TileOperand:
        """Create a TileOperand from parsed info."""
        element_type = ElementType(dtype)
        tile_type = TileType(TileShape(rows, cols), element_type)
        return TileOperand(name, tile_type)
    
    def _create_scalar_operand(self, name: str, dtype: str = "f32") -> ScalarOperand:
        """Create a ScalarOperand."""
        element_type = ElementType(dtype)
        return ScalarOperand(name, element_type)
    
    def _generate_instruction(self, instr: ParsedInstruction) -> List[str]:
        """Generate ARM64 code for a single instruction."""
        lines = []
        indent = self._get_indent()
        
        if instr.opcode == "TILE_DECL":
            # Generate tile declaration
            name = instr.dst
            rows, cols, dtype = instr.operands
            c_type = ARM64_TYPE_MAP.get(dtype, "float")
            lines.append(f"{indent}{c_type} {name}[{rows}][{cols}];")
            return lines
        
        if instr.opcode == "SCALAR_DECL":
            # Generate scalar declaration
            name = instr.dst
            dtype = instr.operands[0]
            c_type = ARM64_TYPE_MAP.get(dtype, "float")
            lines.append(f"{indent}{c_type} {name};")
            return lines
        
        if instr.opcode == "FOR":
            # Generate FOR loop
            iv, lb, ub, step = instr.operands
            lines.append(f"{indent}for (int {iv} = {lb}; {iv} < {ub}; {iv} += {step}) {{")
            self.indent_level += 1
            return lines
        
        if instr.opcode == "ENDFOR":
            self.indent_level = max(0, self.indent_level - 1)
            indent = self._get_indent()
            lines.append(f"{indent}}}")
            return lines
        
        # Get tile info for dimension-dependent operations
        dst_info = self.parser.get_tile_info(instr.dst) if instr.dst else None
        
        # Default dimensions if tile not found
        rows = dst_info.rows if dst_info else 4
        cols = dst_info.cols if dst_info else 4
        dtype = dst_info.dtype if dst_info else "f32"
        
        # Use new instruction-based codegen if enabled
        if self.use_instruction_codegen:
            return self._generate_with_instruction_codegen(instr, rows, cols, dtype)
        else:
            return self._generate_with_legacy_codegen(instr, rows, cols, dtype)
    
    def _generate_with_instruction_codegen(self, instr: ParsedInstruction, 
                                            rows: int, cols: int, dtype: str) -> List[str]:
        """
        Generate ARM64 code using codegen_arm64() method on instruction objects.
        
        This is the new architecture-agnostic approach where each PTO instruction
        class defines its own codegen_arm64() method.
        """
        lines = []
        indent = self._get_indent()
        
        # Update context
        self.arm64_ctx.indent_level = self.indent_level
        
        # Create instruction object and call codegen_arm64()
        pto_instr = None
        
        if instr.opcode == "TADD":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src0 = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            src1 = self._create_tile_operand(instr.operands[1], rows, cols, dtype)
            pto_instr = TADD(dst, src0, src1)
        
        elif instr.opcode == "TSUB":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src0 = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            src1 = self._create_tile_operand(instr.operands[1], rows, cols, dtype)
            pto_instr = TSUB(dst, src0, src1)
        
        elif instr.opcode == "TMUL":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src0 = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            src1 = self._create_tile_operand(instr.operands[1], rows, cols, dtype)
            pto_instr = TMUL(dst, src0, src1)
        
        elif instr.opcode == "TDIV":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src0 = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            src1 = self._create_tile_operand(instr.operands[1], rows, cols, dtype)
            pto_instr = TDIV(dst, src0, src1)
        
        elif instr.opcode == "TMAX":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src0 = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            src1 = self._create_tile_operand(instr.operands[1], rows, cols, dtype)
            pto_instr = TMAX(dst, src0, src1)
        
        elif instr.opcode == "TMIN":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src0 = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            src1 = self._create_tile_operand(instr.operands[1], rows, cols, dtype)
            pto_instr = TMIN(dst, src0, src1)
        
        elif instr.opcode == "TADDS":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            scalar = self._create_scalar_operand(instr.operands[1], dtype)
            pto_instr = TADDS(dst, src, scalar)
        
        elif instr.opcode == "TMULS":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            scalar = self._create_scalar_operand(instr.operands[1], dtype)
            pto_instr = TMULS(dst, src, scalar)
        
        elif instr.opcode == "TDIVS":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            scalar = self._create_scalar_operand(instr.operands[1], dtype)
            pto_instr = TDIVS(dst, src, scalar)
        
        elif instr.opcode == "TABS":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TABS(dst, src)
        
        elif instr.opcode == "TNEG":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TNEG(dst, src)
        
        elif instr.opcode == "TRECIP":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TRECIP(dst, src)
        
        elif instr.opcode == "TEXP":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TEXP(dst, src)
        
        elif instr.opcode == "TLOG":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TLOG(dst, src)
        
        elif instr.opcode == "TSQRT":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TSQRT(dst, src)
        
        elif instr.opcode == "TRSQRT":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TRSQRT(dst, src)
        
        elif instr.opcode == "TRELU":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src = self._create_tile_operand(instr.operands[0], rows, cols, dtype)
            pto_instr = TRELU(dst, src)
        
        elif instr.opcode == "TEXPANDS":
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            scalar = self._create_scalar_operand(instr.operands[0], dtype)
            pto_instr = TEXPANDS(dst, scalar)
        
        elif instr.opcode == "TLOAD":
            src_mem_name, row, col = instr.operands
            dst = self._create_tile_operand(instr.dst, rows, cols, dtype)
            src_mem = MemRefOperand(src_mem_name, MemRefType(MemorySpace.GM, ElementType(dtype)))
            pto_instr = TLOAD(dst, src_mem, ImmediateOperand(int(row)), ImmediateOperand(int(col)))
        
        elif instr.opcode == "TSTORE":
            src_name, row, col = instr.operands
            src_info = self.parser.get_tile_info(src_name)
            if src_info:
                rows, cols = src_info.rows, src_info.cols
            src = self._create_tile_operand(src_name, rows, cols, dtype)
            dst_mem = MemRefOperand(instr.dst, MemRefType(MemorySpace.GM, ElementType(dtype)))
            pto_instr = TSTORE(src, dst_mem, ImmediateOperand(int(row)), ImmediateOperand(int(col)))
        
        elif instr.opcode == "TMATMUL":
            a_name, b_name = instr.operands
            a_info = self.parser.get_tile_info(a_name)
            b_info = self.parser.get_tile_info(b_name)
            m = a_info.rows if a_info else 4
            k = a_info.cols if a_info else 4
            n = b_info.cols if b_info else 4
            dst = self._create_tile_operand(instr.dst, m, n, dtype)
            a = self._create_tile_operand(a_name, m, k, dtype)
            b = self._create_tile_operand(b_name, k, n, dtype)
            pto_instr = TMATMUL(dst, a, b)
        
        elif instr.opcode == "TROWSUM":
            src_name = instr.operands[0]
            src_info = self.parser.get_tile_info(src_name)
            if src_info:
                rows, cols = src_info.rows, src_info.cols
            dst = self._create_tile_operand(instr.dst, rows, 1, dtype)
            src = self._create_tile_operand(src_name, rows, cols, dtype)
            pto_instr = TROWSUM(dst, src)
        
        elif instr.opcode == "TCOLSUM":
            src_name = instr.operands[0]
            src_info = self.parser.get_tile_info(src_name)
            if src_info:
                rows, cols = src_info.rows, src_info.cols
            dst = self._create_tile_operand(instr.dst, 1, cols, dtype)
            src = self._create_tile_operand(src_name, rows, cols, dtype)
            pto_instr = TCOLSUM(dst, src)
        
        # Call codegen_arm64() if instruction was created
        if pto_instr is not None:
            return pto_instr.codegen_arm64(self.arm64_ctx)
        
        # Unknown instruction, add as comment
        lines.append(f"{indent}// Unknown PTO instruction: {instr.opcode}")
        lines.append(f"{indent}// {instr.raw_line}")
        return lines
    
    def _generate_with_legacy_codegen(self, instr: ParsedInstruction,
                                       rows: int, cols: int, dtype: str) -> List[str]:
        """
        Generate ARM64 code using the legacy ARM64CodeGen class.
        
        This maintains backward compatibility with the original codegen approach.
        """
        lines = []
        indent = self._get_indent()
        
        self.codegen.ctx.indent_level = self.indent_level
        
        # Generate based on opcode
        if instr.opcode == "TADD":
            return self.codegen.gen_tadd(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
        
        elif instr.opcode == "TSUB":
            return self.codegen.gen_tsub(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
        
        elif instr.opcode == "TMUL":
            return self.codegen.gen_tmul(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
        
        elif instr.opcode == "TDIV":
            return self.codegen.gen_tdiv(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
        
        elif instr.opcode == "TADDS":
            return self.codegen.gen_tadds(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
        
        elif instr.opcode == "TSUBS":
            # Implement as add with negated scalar
            scalar = instr.operands[1]
            return self.codegen.gen_tadds(instr.dst, instr.operands[0], f"(-{scalar})", rows, cols, dtype)
        
        elif instr.opcode == "TMULS":
            return self.codegen.gen_tmuls(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
        
        elif instr.opcode == "TDIVS":
            return self.codegen.gen_tdivs(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
        
        elif instr.opcode == "TABS":
            return self.codegen.gen_tabs(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TNEG":
            return self.codegen.gen_tneg(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TRECIP":
            return self.codegen.gen_trecip(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TEXP":
            return self.codegen.gen_texp(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TLOG":
            return self.codegen.gen_tlog(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TSQRT":
            return self.codegen.gen_tsqrt(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TRSQRT":
            return self.codegen.gen_trsqrt(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TRELU":
            return self.codegen.gen_trelu(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TEXPANDS":
            return self.codegen.gen_texpands(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TLOAD":
            src, row, col = instr.operands
            src_info = self.parser.get_tile_info(src)
            if src_info:
                rows, cols = src_info.rows, src_info.cols
            return self.codegen.gen_tload(instr.dst, src, int(row), int(col), rows, cols, dtype)
        
        elif instr.opcode == "TSTORE":
            src, row, col = instr.operands
            src_info = self.parser.get_tile_info(src)
            if src_info:
                rows, cols = src_info.rows, src_info.cols
            return self.codegen.gen_tstore(src, instr.dst, int(row), int(col), rows, cols, dtype)
        
        elif instr.opcode == "TMATMUL":
            a, b = instr.operands
            a_info = self.parser.get_tile_info(a)
            b_info = self.parser.get_tile_info(b)
            m = a_info.rows if a_info else 4
            k = a_info.cols if a_info else 4
            n = b_info.cols if b_info else 4
            return self.codegen.gen_tmatmul(instr.dst, a, b, m, k, n, dtype)
        
        elif instr.opcode == "TROWSUM":
            src_info = self.parser.get_tile_info(instr.operands[0])
            if src_info:
                rows, cols = src_info.rows, src_info.cols
            return self.codegen.gen_trowsum(instr.dst, instr.operands[0], rows, cols, dtype)
        
        elif instr.opcode == "TCOLSUM":
            src_info = self.parser.get_tile_info(instr.operands[0])
            if src_info:
                rows, cols = src_info.rows, src_info.cols
            return self.codegen.gen_tcolsum(instr.dst, instr.operands[0], rows, cols, dtype)
        
        else:
            # Unknown instruction, add as comment
            lines.append(f"{indent}// Unknown PTO instruction: {instr.opcode}")
            lines.append(f"{indent}// {instr.raw_line}")
        
        return lines
    
    def compile_file(self, input_path: str, output_path: str):
        """
        Compile a PTO C file to ARM64 C file.
        
        Args:
            input_path: Path to input .c file
            output_path: Path to output .c file
        """
        with open(input_path, 'r') as f:
            input_code = f.read()
        
        output_code = self.compile(input_code)
        
        with open(output_path, 'w') as f:
            f.write(output_code)
        
        print(f"Compiled {input_path} -> {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python pto_c_compiler.py <input.c> <output.c> [--no-fusion]")
        print("\nCompiles PTO C code to ARM64 NEON C code.")
        print("\nOptions:")
        print("  --no-fusion    Disable loop fusion optimization")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Check for --no-fusion flag
    enable_fusion = "--no-fusion" not in sys.argv
    
    compiler = PTOToARM64Compiler(enable_fusion=enable_fusion)
    compiler.compile_file(input_path, output_path)
    
    if enable_fusion:
        print("  (Loop fusion enabled)")
    else:
        print("  (Loop fusion disabled)")


if __name__ == "__main__":
    main()
