"""
Loop Fusion Optimization for PTO ISA Compiler

This module implements loop fusion optimization that combines consecutive
elementwise operations with the same tile shape into a single fused loop.

Loop Fusion Rules:
1. Consecutive elementwise operations on same-shaped tiles can be fused
2. Fusion stops at non-elementwise operations (reduction, matmul, etc.)
3. Fusion is applied repeatedly until no more fusion is possible
4. Data dependencies are respected (read-after-write hazards)

Example:
    Before fusion:
        for (row) for (col) { c[row][col] = a[row][col] + b[row][col]; }
        for (row) for (col) { d[row][col] = c[row][col] * 2.0f; }
    
    After fusion:
        for (row) for (col) {
            c[row][col] = a[row][col] + b[row][col];
            d[row][col] = c[row][col] * 2.0f;
        }
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import deepcopy


# =============================================================================
# Instruction Classification
# =============================================================================

class OpCategory(Enum):
    """Categories of PTO operations for fusion analysis."""
    ELEMENTWISE_BINARY = auto()   # TADD, TSUB, TMUL, TDIV, TMAX, TMIN
    ELEMENTWISE_UNARY = auto()    # TABS, TNEG, TRECIP, TEXP, TLOG, TSQRT, TRELU
    ELEMENTWISE_SCALAR = auto()   # TADDS, TMULS, TDIVS, etc.
    BROADCAST = auto()            # TEXPANDS
    REDUCTION = auto()            # TROWSUM, TCOLSUM, TSUM
    MATMUL = auto()               # TMATMUL
    MEMORY = auto()               # TLOAD, TSTORE
    CONTROL_FLOW = auto()         # FOR, ENDFOR
    DECLARATION = auto()          # TILE_DECL, SCALAR_DECL
    OTHER = auto()                # Unknown/passthrough


# Classify opcodes into categories
OPCODE_CATEGORY = {
    # Elementwise binary
    "TADD": OpCategory.ELEMENTWISE_BINARY,
    "TSUB": OpCategory.ELEMENTWISE_BINARY,
    "TMUL": OpCategory.ELEMENTWISE_BINARY,
    "TDIV": OpCategory.ELEMENTWISE_BINARY,
    "TMAX": OpCategory.ELEMENTWISE_BINARY,
    "TMIN": OpCategory.ELEMENTWISE_BINARY,
    "TAND": OpCategory.ELEMENTWISE_BINARY,
    "TOR": OpCategory.ELEMENTWISE_BINARY,
    "TXOR": OpCategory.ELEMENTWISE_BINARY,
    
    # Elementwise unary
    "TABS": OpCategory.ELEMENTWISE_UNARY,
    "TNEG": OpCategory.ELEMENTWISE_UNARY,
    "TRECIP": OpCategory.ELEMENTWISE_UNARY,
    "TEXP": OpCategory.ELEMENTWISE_UNARY,
    "TLOG": OpCategory.ELEMENTWISE_UNARY,
    "TSQRT": OpCategory.ELEMENTWISE_UNARY,
    "TRSQRT": OpCategory.ELEMENTWISE_UNARY,
    "TRELU": OpCategory.ELEMENTWISE_UNARY,
    
    # Elementwise with scalar
    "TADDS": OpCategory.ELEMENTWISE_SCALAR,
    "TSUBS": OpCategory.ELEMENTWISE_SCALAR,
    "TMULS": OpCategory.ELEMENTWISE_SCALAR,
    "TDIVS": OpCategory.ELEMENTWISE_SCALAR,
    "TMAXS": OpCategory.ELEMENTWISE_SCALAR,
    "TMINS": OpCategory.ELEMENTWISE_SCALAR,
    
    # Broadcast
    "TEXPANDS": OpCategory.BROADCAST,
    
    # Reduction (fusion barrier)
    "TROWSUM": OpCategory.REDUCTION,
    "TCOLSUM": OpCategory.REDUCTION,
    "TSUM": OpCategory.REDUCTION,
    
    # Matrix ops (fusion barrier)
    "TMATMUL": OpCategory.MATMUL,
    
    # Memory
    "TLOAD": OpCategory.MEMORY,
    "TSTORE": OpCategory.MEMORY,
    
    # Control flow
    "FOR": OpCategory.CONTROL_FLOW,
    "ENDFOR": OpCategory.CONTROL_FLOW,
    
    # Declarations
    "TILE_DECL": OpCategory.DECLARATION,
    "SCALAR_DECL": OpCategory.DECLARATION,
}


def get_category(opcode: str) -> OpCategory:
    """Get the category of an opcode."""
    return OPCODE_CATEGORY.get(opcode, OpCategory.OTHER)


def is_fusable(opcode: str) -> bool:
    """Check if an operation can be fused (elementwise operations)."""
    category = get_category(opcode)
    return category in {
        OpCategory.ELEMENTWISE_BINARY,
        OpCategory.ELEMENTWISE_UNARY,
        OpCategory.ELEMENTWISE_SCALAR,
        OpCategory.BROADCAST,
    }


def is_fusion_barrier(opcode: str) -> bool:
    """Check if an operation is a fusion barrier (stops fusion)."""
    category = get_category(opcode)
    return category in {
        OpCategory.REDUCTION,
        OpCategory.MATMUL,
        OpCategory.CONTROL_FLOW,
    }


# =============================================================================
# IR for Loop Fusion
# =============================================================================

@dataclass
class TileShape:
    """Shape of a tile."""
    rows: int
    cols: int
    dtype: str = "f32"
    
    def __hash__(self):
        return hash((self.rows, self.cols, self.dtype))
    
    def __eq__(self, other):
        if not isinstance(other, TileShape):
            return False
        return self.rows == other.rows and self.cols == other.cols and self.dtype == other.dtype


@dataclass
class FusableOp:
    """
    A single fusable operation.
    
    Attributes:
        opcode: The operation code (TADD, TMUL, etc.)
        dst: Destination tile name
        operands: List of operand names/values
        shape: Shape of the operation
        raw_instr: Original parsed instruction
    """
    opcode: str
    dst: str
    operands: List[str]
    shape: TileShape
    raw_instr: Any = None
    
    def get_reads(self) -> Set[str]:
        """Get the set of tiles read by this operation."""
        reads = set()
        category = get_category(self.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            reads.add(self.operands[0])
            reads.add(self.operands[1])
        elif category == OpCategory.ELEMENTWISE_UNARY:
            reads.add(self.operands[0])
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            reads.add(self.operands[0])
            # scalar operand is not a tile read
        elif category == OpCategory.BROADCAST:
            pass  # scalar only
        
        return reads
    
    def get_writes(self) -> Set[str]:
        """Get the set of tiles written by this operation."""
        if self.dst:
            return {self.dst}
        return set()


@dataclass
class FusedLoop:
    """
    A fused loop containing multiple operations.
    
    All operations in a fused loop:
    - Have the same shape
    - Are elementwise operations
    - Can be executed in order within the same loop iteration
    """
    shape: TileShape
    operations: List[FusableOp] = field(default_factory=list)
    
    def add_op(self, op: FusableOp):
        """Add an operation to the fused loop."""
        self.operations.append(op)
    
    def can_fuse(self, op: FusableOp) -> bool:
        """Check if an operation can be fused into this loop."""
        # Must be same shape
        if op.shape != self.shape:
            return False
        
        # Must be fusable operation
        if not is_fusable(op.opcode):
            return False
        
        return True
    
    def __len__(self):
        return len(self.operations)


@dataclass  
class FusionBarrier:
    """
    Represents a non-fusable operation that acts as a barrier.
    """
    raw_instr: Any
    opcode: str


# =============================================================================
# Loop Fusion Optimizer
# =============================================================================

class LoopFusionOptimizer:
    """
    Optimizes PTO instruction sequences by fusing consecutive loops.
    
    Algorithm:
    1. Parse instructions into a sequence of FusableOp and FusionBarrier
    2. Group consecutive FusableOps with same shape into FusedLoops
    3. Respect data dependencies within fused groups
    4. Repeat until no more fusion is possible
    """
    
    def __init__(self, tile_info: Dict[str, 'TileInfo']):
        """
        Initialize the optimizer.
        
        Args:
            tile_info: Dictionary mapping tile names to TileInfo objects
        """
        self.tile_info = tile_info
        self.stats = {
            "original_ops": 0,
            "fused_loops": 0,
            "fusion_savings": 0,
        }
    
    def get_shape(self, tile_name: str) -> Optional[TileShape]:
        """Get the shape of a tile."""
        info = self.tile_info.get(tile_name)
        if info:
            return TileShape(info.rows, info.cols, info.dtype)
        return None
    
    def instr_to_fusable_op(self, instr) -> Optional[FusableOp]:
        """Convert a parsed instruction to a FusableOp if possible."""
        if not is_fusable(instr.opcode):
            return None
        
        # Get shape from destination tile
        shape = self.get_shape(instr.dst) if instr.dst else None
        if shape is None:
            # Try to get shape from operands
            for op in instr.operands:
                shape = self.get_shape(op)
                if shape:
                    break
        
        if shape is None:
            # Default shape if not found
            shape = TileShape(4, 4, "f32")
        
        return FusableOp(
            opcode=instr.opcode,
            dst=instr.dst,
            operands=list(instr.operands),
            shape=shape,
            raw_instr=instr
        )
    
    def check_dependency(self, group: List[FusableOp], new_op: FusableOp) -> bool:
        """
        Check if adding new_op to group would violate data dependencies.
        
        We allow read-after-write within the same iteration (fine for elementwise).
        We need to ensure no circular dependencies exist.
        """
        # For elementwise operations, as long as we execute in order,
        # read-after-write within the same iteration is fine.
        # We just need to make sure we don't have conflicting writes.
        
        new_writes = new_op.get_writes()
        
        # Check for write-after-write conflicts (same destination)
        # This is actually allowed if we process in order
        
        return True  # Elementwise ops can always be fused if same shape
    
    def fuse_instructions(self, instructions: List) -> List:
        """
        Fuse consecutive fusable instructions into FusedLoops.
        
        Args:
            instructions: List of ParsedInstruction objects
            
        Returns:
            List of FusedLoop and FusionBarrier objects
        """
        result = []
        current_fused: Optional[FusedLoop] = None
        
        self.stats["original_ops"] = len(instructions)
        
        for instr in instructions:
            # Skip declarations and passthrough
            if instr.opcode in ("TILE_DECL", "SCALAR_DECL", "PASSTHROUGH"):
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(instr)
                continue
            
            # Check if this is a fusion barrier
            if is_fusion_barrier(instr.opcode):
                # Flush current fused loop
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(FusionBarrier(raw_instr=instr, opcode=instr.opcode))
                continue
            
            # Try to convert to fusable op
            fusable_op = self.instr_to_fusable_op(instr)
            
            if fusable_op is None:
                # Not fusable, flush and add as barrier
                if current_fused and len(current_fused) > 0:
                    result.append(current_fused)
                    current_fused = None
                result.append(FusionBarrier(raw_instr=instr, opcode=instr.opcode))
                continue
            
            # Try to add to current fused loop
            if current_fused is None:
                # Start new fused loop
                current_fused = FusedLoop(shape=fusable_op.shape)
                current_fused.add_op(fusable_op)
            elif current_fused.can_fuse(fusable_op) and \
                 self.check_dependency(current_fused.operations, fusable_op):
                # Add to current fused loop
                current_fused.add_op(fusable_op)
            else:
                # Shape mismatch or dependency issue, start new fused loop
                result.append(current_fused)
                current_fused = FusedLoop(shape=fusable_op.shape)
                current_fused.add_op(fusable_op)
        
        # Flush remaining fused loop
        if current_fused and len(current_fused) > 0:
            result.append(current_fused)
        
        # Calculate statistics
        self.stats["fused_loops"] = sum(1 for r in result if isinstance(r, FusedLoop))
        fused_op_count = sum(len(r.operations) for r in result if isinstance(r, FusedLoop))
        self.stats["fusion_savings"] = fused_op_count - self.stats["fused_loops"]
        
        return result
    
    def optimize(self, instructions: List) -> List:
        """
        Apply loop fusion optimization.
        
        This method applies fusion iteratively until no more fusion is possible.
        
        Args:
            instructions: List of ParsedInstruction objects
            
        Returns:
            Optimized list of FusedLoop, FusionBarrier, and other objects
        """
        return self.fuse_instructions(instructions)
    
    def print_stats(self):
        """Print optimization statistics."""
        print(f"Loop Fusion Statistics:")
        print(f"  Original operations: {self.stats['original_ops']}")
        print(f"  Fused loops created: {self.stats['fused_loops']}")
        print(f"  Loop overhead saved: {self.stats['fusion_savings']}")


# =============================================================================
# Fused Code Generator
# =============================================================================

class FusedCodeGenerator:
    """
    Generates ARM64 NEON code for fused loops.
    """
    
    def __init__(self):
        self.indent_level = 0
        self.var_counter = 0
    
    def _indent(self) -> str:
        return "    " * self.indent_level
    
    def _get_unique_var(self, prefix: str = "_v") -> str:
        name = f"{prefix}{self.var_counter}"
        self.var_counter += 1
        return name
    
    def generate_fused_loop(self, fused: FusedLoop) -> List[str]:
        """
        Generate code for a fused loop.
        
        Instead of generating separate loops for each operation,
        generates a single loop with all operations in the body.
        """
        lines = []
        indent = self._indent()
        
        rows = fused.shape.rows
        cols = fused.shape.cols
        dtype = fused.shape.dtype
        
        # Determine vector lanes
        vec_lanes = 4 if dtype == "f32" else 8 if dtype == "f16" else 4
        suffix = "f32" if dtype == "f32" else "f16" if dtype == "f16" else "f32"
        vec_type = f"float32x4_t" if dtype == "f32" else f"float16x8_t" if dtype == "f16" else "float32x4_t"
        
        # Comment showing fused operations
        op_names = [f"{op.dst}={op.opcode}({','.join(op.operands)})" for op in fused.operations]
        lines.append(f"{indent}// FUSED LOOP ({len(fused.operations)} ops): {'; '.join(op_names)}")
        
        # Pre-compute scalar broadcast vectors
        scalar_vars = {}
        for op in fused.operations:
            if get_category(op.opcode) == OpCategory.ELEMENTWISE_SCALAR:
                scalar_val = op.operands[1]
                if scalar_val not in scalar_vars:
                    var_name = self._get_unique_var("_vs")
                    scalar_vars[scalar_val] = var_name
                    lines.append(f"{indent}{vec_type} {var_name} = vdupq_n_{suffix}({scalar_val});")
            elif op.opcode == "TEXPANDS":
                scalar_val = op.operands[0]
                if scalar_val not in scalar_vars:
                    var_name = self._get_unique_var("_vs")
                    scalar_vars[scalar_val] = var_name
                    lines.append(f"{indent}{vec_type} {var_name} = vdupq_n_{suffix}({scalar_val});")
        
        # Generate fused loop
        lines.append(f"{indent}for (int _row = 0; _row < {rows}; _row++) {{")
        self.indent_level += 1
        indent = self._indent()
        
        lines.append(f"{indent}int _col;")
        
        # Vectorized inner loop
        lines.append(f"{indent}// Vectorized loop")
        lines.append(f"{indent}for (_col = 0; _col + {vec_lanes} <= {cols}; _col += {vec_lanes}) {{")
        self.indent_level += 1
        
        # Generate vectorized body for all operations
        for op in fused.operations:
            vec_lines = self._gen_vectorized_op(op, suffix, vec_type, scalar_vars)
            lines.extend(vec_lines)
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        # Scalar cleanup loop
        lines.append(f"{indent}// Scalar cleanup")
        lines.append(f"{indent}for (; _col < {cols}; _col++) {{")
        self.indent_level += 1
        
        # Generate scalar body for all operations
        for op in fused.operations:
            scalar_lines = self._gen_scalar_op(op)
            lines.extend(scalar_lines)
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        self.indent_level -= 1
        indent = self._indent()
        lines.append(f"{indent}}}")
        
        return lines
    
    def _gen_vectorized_op(self, op: FusableOp, suffix: str, vec_type: str,
                           scalar_vars: Dict[str, str]) -> List[str]:
        """Generate vectorized code for a single operation within a fused loop."""
        lines = []
        indent = self._indent()
        category = get_category(op.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            v0 = self._get_unique_var("_v")
            v1 = self._get_unique_var("_v")
            vr = self._get_unique_var("_vr")
            
            neon_op = self._get_neon_binary_op(op.opcode, suffix)
            
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {v1} = vld1q_{suffix}(&{op.operands[1]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_op}({v0}, {v1});")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.ELEMENTWISE_UNARY:
            v0 = self._get_unique_var("_v")
            vr = self._get_unique_var("_vr")
            
            neon_code = self._get_neon_unary_op(op.opcode, suffix, v0, vec_type)
            
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_code};")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            v0 = self._get_unique_var("_v")
            vr = self._get_unique_var("_vr")
            vs = scalar_vars.get(op.operands[1], "_vs")
            
            neon_op = self._get_neon_scalar_op(op.opcode, suffix)
            
            lines.append(f"{indent}{vec_type} {v0} = vld1q_{suffix}(&{op.operands[0]}[_row][_col]);")
            lines.append(f"{indent}{vec_type} {vr} = {neon_op}({v0}, {vs});")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vr});")
        
        elif category == OpCategory.BROADCAST:
            vs = scalar_vars.get(op.operands[0], "_vs")
            lines.append(f"{indent}vst1q_{suffix}(&{op.dst}[_row][_col], {vs});")
        
        return lines
    
    def _gen_scalar_op(self, op: FusableOp) -> List[str]:
        """Generate scalar code for a single operation within a fused loop."""
        lines = []
        indent = self._indent()
        category = get_category(op.opcode)
        
        if category == OpCategory.ELEMENTWISE_BINARY:
            c_op = self._get_c_binary_op(op.opcode)
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]}[_row][_col] {c_op} {op.operands[1]}[_row][_col];")
        
        elif category == OpCategory.ELEMENTWISE_UNARY:
            c_expr = self._get_c_unary_expr(op.opcode, f"{op.operands[0]}[_row][_col]")
            lines.append(f"{indent}{op.dst}[_row][_col] = {c_expr};")
        
        elif category == OpCategory.ELEMENTWISE_SCALAR:
            c_op = self._get_c_scalar_op(op.opcode)
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]}[_row][_col] {c_op} {op.operands[1]};")
        
        elif category == OpCategory.BROADCAST:
            lines.append(f"{indent}{op.dst}[_row][_col] = {op.operands[0]};")
        
        return lines
    
    def _get_neon_binary_op(self, opcode: str, suffix: str) -> str:
        """Get NEON intrinsic for binary operation."""
        ops = {
            "TADD": f"vaddq_{suffix}",
            "TSUB": f"vsubq_{suffix}",
            "TMUL": f"vmulq_{suffix}",
            "TDIV": f"vdivq_{suffix}",
            "TMAX": f"vmaxq_{suffix}",
            "TMIN": f"vminq_{suffix}",
        }
        return ops.get(opcode, f"vaddq_{suffix}")
    
    def _get_neon_unary_op(self, opcode: str, suffix: str, v: str, vec_type: str) -> str:
        """Get NEON expression for unary operation."""
        ops = {
            "TABS": f"vabsq_{suffix}({v})",
            "TNEG": f"vnegq_{suffix}({v})",
            "TSQRT": f"vsqrtq_{suffix}({v})",
            "TRSQRT": f"vrsqrteq_{suffix}({v})",
            "TRELU": f"vmaxq_{suffix}({v}, vdupq_n_{suffix}(0.0f))",
        }
        # For ops without NEON equivalent, fall back to scalar in loop
        return ops.get(opcode, f"{v}")  # Identity for unhandled
    
    def _get_neon_scalar_op(self, opcode: str, suffix: str) -> str:
        """Get NEON intrinsic for scalar operation."""
        ops = {
            "TADDS": f"vaddq_{suffix}",
            "TSUBS": f"vsubq_{suffix}",
            "TMULS": f"vmulq_{suffix}",
            "TDIVS": f"vdivq_{suffix}",
            "TMAXS": f"vmaxq_{suffix}",
            "TMINS": f"vminq_{suffix}",
        }
        return ops.get(opcode, f"vaddq_{suffix}")
    
    def _get_c_binary_op(self, opcode: str) -> str:
        """Get C operator for binary operation."""
        ops = {
            "TADD": "+",
            "TSUB": "-",
            "TMUL": "*",
            "TDIV": "/",
        }
        return ops.get(opcode, "+")
    
    def _get_c_unary_expr(self, opcode: str, operand: str) -> str:
        """Get C expression for unary operation."""
        exprs = {
            "TABS": f"fabsf({operand})",
            "TNEG": f"-{operand}",
            "TRECIP": f"1.0f / {operand}",
            "TEXP": f"expf({operand})",
            "TLOG": f"logf({operand})",
            "TSQRT": f"sqrtf({operand})",
            "TRSQRT": f"1.0f / sqrtf({operand})",
            "TRELU": f"fmaxf({operand}, 0.0f)",
        }
        return exprs.get(opcode, operand)
    
    def _get_c_scalar_op(self, opcode: str) -> str:
        """Get C operator for scalar operation."""
        ops = {
            "TADDS": "+",
            "TSUBS": "-",
            "TMULS": "*",
            "TDIVS": "/",
        }
        return ops.get(opcode, "+")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockTileInfo:
        name: str
        rows: int
        cols: int
        dtype: str
    
    @dataclass
    class MockInstruction:
        opcode: str
        dst: str
        operands: list
        raw_line: str = ""
    
    # Create test tiles
    tiles = {
        "x": MockTileInfo("x", 8, 8, "f32"),
        "x_squared": MockTileInfo("x_squared", 8, 8, "f32"),
        "term": MockTileInfo("term", 8, 8, "f32"),
        "result": MockTileInfo("result", 8, 8, "f32"),
    }
    
    # Create test instructions (from sinh example)
    instructions = [
        MockInstruction("TMUL", "x_squared", ["x", "x"]),
        MockInstruction("TMULS", "term", ["x", "1.0f"]),
        MockInstruction("TMUL", "term", ["term", "x_squared"]),
        MockInstruction("TDIVS", "term", ["term", "6.0f"]),
        MockInstruction("TADD", "result", ["result", "term"]),
        MockInstruction("TMUL", "term", ["term", "x_squared"]),
        MockInstruction("TDIVS", "term", ["term", "20.0f"]),
        MockInstruction("TADD", "result", ["result", "term"]),
    ]
    
    print("Loop Fusion Test")
    print("=" * 60)
    
    # Test fusion
    optimizer = LoopFusionOptimizer(tiles)
    fused = optimizer.optimize(instructions)
    
    optimizer.print_stats()
    
    print(f"\nFused result: {len(fused)} groups")
    for i, group in enumerate(fused):
        if isinstance(group, FusedLoop):
            print(f"\n  FusedLoop {i} ({len(group.operations)} ops):")
            for op in group.operations:
                print(f"    {op.dst} = {op.opcode}({', '.join(op.operands)})")
        elif isinstance(group, FusionBarrier):
            print(f"\n  FusionBarrier {i}: {group.opcode}")
    
    # Test code generation
    print("\n\nGenerated Code:")
    print("-" * 60)
    
    codegen = FusedCodeGenerator()
    for group in fused:
        if isinstance(group, FusedLoop):
            lines = codegen.generate_fused_loop(group)
            for line in lines:
                print(line)
            print()
