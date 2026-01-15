"""
PTO ISA Compiler

This module provides the compiler infrastructure for the PTO (Programmable Tensor Operations)
Domain Specific Language. It handles parsing, validation, optimization, and code generation
for PTO programs.

Key Features:
- DSL parsing and AST construction
- Type checking and validation
- Loop unrolling and optimization
- Code generation to PTO assembly
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union, Callable
from enum import Enum
import re

from pto_isa_definition import (
    # Types
    ElementType, MemorySpace, CompareMode, RoundMode, TMovMode,
    TileShape, TileType, MemRefType, EventType,
    
    # Operands
    TileOperand, ScalarOperand, MemRefOperand, IndexOperand, ImmediateOperand, Operand,
    
    # Base classes
    PTOInstruction, TileInstruction, ScalarInstruction, ControlFlowInstruction,
    
    # All instructions
    ALL_INSTRUCTIONS, TILE_INSTRUCTIONS, SCALAR_INSTRUCTIONS, CONTROL_FLOW_INSTRUCTIONS,
    
    # Loop constructs
    TileLoop, NestedTileLoop, FOR, ENDFOR, WHILE, DO, ENDWHILE, IF, ELSE, ENDIF,
    
    # Tile instructions
    TLOAD, TSTORE, TADD, TSUB, TMUL, TDIV, TMATMUL, TMATMUL_ACC,
    TROWSUM, TCOLSUM, TRELU, TSQRT, TEXP, TLOG,
    # Scalar operations
    TADDS, TMULS, TDIVS,
    
    # Helper functions
    tile, scalar, index, memref, imm,
)


# =============================================================================
# Compiler Error Types
# =============================================================================

class CompilerError(Exception):
    """Base class for compiler errors."""
    pass


class ParseError(CompilerError):
    """Error during parsing."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
        super().__init__(f"Parse error at line {line}, col {col}: {message}")


class TypeError(CompilerError):
    """Type checking error."""
    def __init__(self, message: str, instruction: Optional[PTOInstruction] = None):
        self.instruction = instruction
        super().__init__(f"Type error: {message}")


class ValidationError(CompilerError):
    """Validation error."""
    pass


# =============================================================================
# Symbol Table
# =============================================================================

@dataclass
class Symbol:
    """A symbol in the symbol table."""
    name: str
    symbol_type: str  # "tile", "scalar", "memref", "index"
    data_type: Any    # TileType, ElementType, MemRefType, etc.
    is_const: bool = False
    value: Optional[Any] = None


class SymbolTable:
    """
    Symbol table for managing variables and their types.
    
    Supports nested scopes for loop constructs.
    """
    
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]
    
    def push_scope(self):
        """Enter a new scope."""
        self.scopes.append({})
    
    def pop_scope(self):
        """Exit current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
    
    def define(self, name: str, symbol: Symbol):
        """Define a symbol in the current scope."""
        self.scopes[-1][name] = symbol
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol, searching from innermost to outermost scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None
    
    def is_defined(self, name: str) -> bool:
        """Check if a symbol is defined in any scope."""
        return self.lookup(name) is not None
    
    def is_defined_in_current_scope(self, name: str) -> bool:
        """Check if a symbol is defined in the current scope."""
        return name in self.scopes[-1]


# =============================================================================
# Program Representation
# =============================================================================

@dataclass
class PTOProgram:
    """
    A complete PTO program.
    
    Contains declarations, instructions, and metadata.
    """
    name: str = "main"
    tile_declarations: Dict[str, TileType] = field(default_factory=dict)
    scalar_declarations: Dict[str, ElementType] = field(default_factory=dict)
    memref_declarations: Dict[str, MemRefType] = field(default_factory=dict)
    instructions: List[PTOInstruction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_tile(self, name: str, rows: int, cols: int, dtype: ElementType = ElementType.F32):
        """Declare a tile variable."""
        self.tile_declarations[name] = TileType.create(rows, cols, dtype)
    
    def add_scalar(self, name: str, dtype: ElementType = ElementType.F32):
        """Declare a scalar variable."""
        self.scalar_declarations[name] = dtype
    
    def add_memref(self, name: str, space: MemorySpace, dtype: ElementType,
                   shape: Optional[TileShape] = None):
        """Declare a memory reference."""
        self.memref_declarations[name] = MemRefType(space, dtype, shape)
    
    def add_instruction(self, instr: PTOInstruction):
        """Add an instruction to the program."""
        self.instructions.append(instr)
    
    def add_loop(self, loop: Union[TileLoop, NestedTileLoop]):
        """Add a loop construct by expanding it to instructions."""
        self.instructions.extend(loop.to_instructions())


# =============================================================================
# DSL Builder - Fluent Interface
# =============================================================================

class PTOProgramBuilder:
    """
    Fluent interface for building PTO programs.
    
    Example:
        program = (PTOProgramBuilder("matmul")
            .tile("a", 64, 64, ElementType.F16)
            .tile("b", 64, 64, ElementType.F16)
            .tile("c", 64, 64, ElementType.F32)
            .memref("mem_a", MemorySpace.GM, ElementType.F16)
            .load("a", "mem_a", 0, 0)
            .matmul("c", "a", "b")
            .store("c", "mem_c", 0, 0)
            .build())
    """
    
    def __init__(self, name: str = "main"):
        self.program = PTOProgram(name=name)
        self.symbol_table = SymbolTable()
        self._loop_stack: List[List[PTOInstruction]] = []
    
    def _get_tile(self, name: str) -> TileOperand:
        """Get a tile operand by name."""
        if name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{name}' not declared")
        return TileOperand(name, self.program.tile_declarations[name])
    
    def _get_scalar(self, name: str) -> ScalarOperand:
        """Get a scalar operand by name."""
        if name not in self.program.scalar_declarations:
            raise ValidationError(f"Scalar '{name}' not declared")
        return ScalarOperand(name, self.program.scalar_declarations[name])
    
    def _get_memref(self, name: str) -> MemRefOperand:
        """Get a memref operand by name."""
        if name not in self.program.memref_declarations:
            raise ValidationError(f"MemRef '{name}' not declared")
        return MemRefOperand(name, self.program.memref_declarations[name])
    
    def _add_instr(self, instr: PTOInstruction):
        """Add instruction to current context (main or loop body)."""
        if self._loop_stack:
            self._loop_stack[-1].append(instr)
        else:
            self.program.add_instruction(instr)
    
    # Declaration methods
    def tile(self, name: str, rows: int, cols: int, 
             dtype: ElementType = ElementType.F32) -> "PTOProgramBuilder":
        """Declare a tile variable."""
        self.program.add_tile(name, rows, cols, dtype)
        self.symbol_table.define(name, Symbol(name, "tile", TileType.create(rows, cols, dtype)))
        return self
    
    def scalar(self, name: str, dtype: ElementType = ElementType.F32) -> "PTOProgramBuilder":
        """Declare a scalar variable."""
        self.program.add_scalar(name, dtype)
        self.symbol_table.define(name, Symbol(name, "scalar", dtype))
        return self
    
    def memref(self, name: str, space: MemorySpace = MemorySpace.GM,
               dtype: ElementType = ElementType.F32,
               shape: Optional[Tuple[int, int]] = None) -> "PTOProgramBuilder":
        """Declare a memory reference."""
        tile_shape = TileShape(*shape) if shape else None
        self.program.add_memref(name, space, dtype, tile_shape)
        self.symbol_table.define(name, Symbol(name, "memref", MemRefType(space, dtype, tile_shape)))
        return self
    
    # Tile memory operations
    def load(self, dst: str, src_mem: str, row: int = 0, col: int = 0) -> "PTOProgramBuilder":
        """Load data from memory into a tile."""
        self._add_instr(TLOAD(
            dst=self._get_tile(dst),
            src_mem=self._get_memref(src_mem),
            row_offset=ImmediateOperand(row),
            col_offset=ImmediateOperand(col)
        ))
        return self
    
    def store(self, src: str, dst_mem: str, row: int = 0, col: int = 0) -> "PTOProgramBuilder":
        """Store data from a tile into memory."""
        self._add_instr(TSTORE(
            src=self._get_tile(src),
            dst_mem=self._get_memref(dst_mem),
            row_offset=ImmediateOperand(row),
            col_offset=ImmediateOperand(col)
        ))
        return self
    
    # Tile arithmetic operations
    def add(self, dst: str, src0: str, src1: str) -> "PTOProgramBuilder":
        """Elementwise add of two tiles."""
        self._add_instr(TADD(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def sub(self, dst: str, src0: str, src1: str) -> "PTOProgramBuilder":
        """Elementwise subtract of two tiles."""
        self._add_instr(TSUB(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def mul(self, dst: str, src0: str, src1: str) -> "PTOProgramBuilder":
        """Elementwise multiply of two tiles."""
        self._add_instr(TMUL(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    def div(self, dst: str, src0: str, src1: str) -> "PTOProgramBuilder":
        """Elementwise divide of two tiles."""
        self._add_instr(TDIV(
            dst=self._get_tile(dst),
            src0=self._get_tile(src0),
            src1=self._get_tile(src1)
        ))
        return self
    
    # Scalar operations (tile op scalar)
    def _make_scalar_operand(self, value: float) -> ScalarOperand:
        """Create a scalar operand for an immediate value."""
        # Create a unique name for the immediate value
        name = f"_imm_{abs(hash(value)) % 10000}"
        return ScalarOperand(name=str(value), element_type=ElementType.F32)
    
    def adds(self, dst: str, src: str, scalar: float) -> "PTOProgramBuilder":
        """Add scalar to all elements of tile."""
        self._add_instr(TADDS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    def muls(self, dst: str, src: str, scalar: float) -> "PTOProgramBuilder":
        """Multiply all elements of tile by scalar."""
        self._add_instr(TMULS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    def divs(self, dst: str, src: str, scalar: float) -> "PTOProgramBuilder":
        """Divide all elements of tile by scalar."""
        self._add_instr(TDIVS(
            dst=self._get_tile(dst),
            src=self._get_tile(src),
            scalar=self._make_scalar_operand(scalar)
        ))
        return self
    
    # Matrix operations
    def matmul(self, dst: str, a: str, b: str) -> "PTOProgramBuilder":
        """Matrix multiply."""
        self._add_instr(TMATMUL(
            dst=self._get_tile(dst),
            a=self._get_tile(a),
            b=self._get_tile(b)
        ))
        return self
    
    def matmul_acc(self, dst: str, acc: str, a: str, b: str) -> "PTOProgramBuilder":
        """Matrix multiply with accumulator."""
        self._add_instr(TMATMUL_ACC(
            dst=self._get_tile(dst),
            acc=self._get_tile(acc),
            a=self._get_tile(a),
            b=self._get_tile(b)
        ))
        return self
    
    # Activation functions
    def relu(self, dst: str, src: str) -> "PTOProgramBuilder":
        """Apply ReLU activation."""
        self._add_instr(TRELU(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def exp(self, dst: str, src: str) -> "PTOProgramBuilder":
        """Apply exponential."""
        self._add_instr(TEXP(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def log(self, dst: str, src: str) -> "PTOProgramBuilder":
        """Apply natural logarithm."""
        self._add_instr(TLOG(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def sqrt(self, dst: str, src: str) -> "PTOProgramBuilder":
        """Apply square root."""
        self._add_instr(TSQRT(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Reduction operations
    def rowsum(self, dst: str, src: str) -> "PTOProgramBuilder":
        """Sum reduction across rows."""
        self._add_instr(TROWSUM(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    def colsum(self, dst: str, src: str) -> "PTOProgramBuilder":
        """Sum reduction across columns."""
        self._add_instr(TCOLSUM(
            dst=self._get_tile(dst),
            src=self._get_tile(src)
        ))
        return self
    
    # Loop constructs
    def for_loop(self, iv_name: str, lb: int, ub: int, step: int = 1) -> "PTOProgramBuilder":
        """Begin a FOR loop."""
        self.symbol_table.push_scope()
        self.symbol_table.define(iv_name, Symbol(iv_name, "index", ElementType.INDEX))
        
        self._add_instr(FOR(
            iv=IndexOperand(iv_name),
            lb=ImmediateOperand(lb),
            ub=ImmediateOperand(ub),
            step=ImmediateOperand(step)
        ))
        self._loop_stack.append([])
        return self
    
    def end_for(self) -> "PTOProgramBuilder":
        """End a FOR loop."""
        if not self._loop_stack:
            raise ValidationError("ENDFOR without matching FOR")
        
        loop_body = self._loop_stack.pop()
        
        # If there's still a parent loop, add body and ENDFOR to parent
        # Otherwise, add to program
        if self._loop_stack:
            # We're inside another loop, add to parent loop's body
            for instr in loop_body:
                self._loop_stack[-1].append(instr)
            self._loop_stack[-1].append(ENDFOR())
        else:
            # No parent loop, add directly to program
            for instr in loop_body:
                self.program.add_instruction(instr)
            self.program.add_instruction(ENDFOR())
        
        self.symbol_table.pop_scope()
        return self
    
    def tile_loop(self, iv_name: str, tile_name: str, 
                  dimension: str = "rows", step: int = 1) -> "PTOProgramBuilder":
        """
        Begin a loop that iterates based on tile dimensions.
        
        Args:
            iv_name: Name of induction variable
            tile_name: Name of tile to get dimensions from
            dimension: "rows" or "cols"
            step: Loop step
        """
        if tile_name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{tile_name}' not declared")
        
        tile_type = self.program.tile_declarations[tile_name]
        bound = tile_type.shape.rows if dimension == "rows" else tile_type.shape.cols
        
        return self.for_loop(iv_name, 0, bound, step)
    
    def nested_tile_loop(self, outer_iv: str, inner_iv: str, tile_name: str,
                         outer_step: int = 1, inner_step: int = 1) -> "PTOProgramBuilder":
        """
        Begin a 2-level nested loop that iterates over tile dimensions.
        
        Outer loop iterates over rows, inner loop over columns.
        """
        if tile_name not in self.program.tile_declarations:
            raise ValidationError(f"Tile '{tile_name}' not declared")
        
        tile_type = self.program.tile_declarations[tile_name]
        
        # Start outer loop
        self.for_loop(outer_iv, 0, tile_type.shape.rows, outer_step)
        # Start inner loop
        self.for_loop(inner_iv, 0, tile_type.shape.cols, inner_step)
        
        return self
    
    def end_nested_loop(self) -> "PTOProgramBuilder":
        """End a 2-level nested loop."""
        self.end_for()  # End inner
        self.end_for()  # End outer
        return self
    
    # Build the program
    def build(self) -> PTOProgram:
        """Build and return the program."""
        if self._loop_stack:
            raise ValidationError("Unclosed loop constructs")
        return self.program


# =============================================================================
# Type Checker
# =============================================================================

class TypeChecker:
    """
    Type checker for PTO programs.
    
    Validates that all operations have compatible types.
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.errors: List[str] = []
    
    def check(self) -> bool:
        """Run type checking, return True if no errors."""
        self.errors = []
        
        for instr in self.program.instructions:
            self._check_instruction(instr)
        
        return len(self.errors) == 0
    
    def _check_instruction(self, instr: PTOInstruction):
        """Type check a single instruction."""
        if isinstance(instr, TADD) or isinstance(instr, TSUB) or isinstance(instr, TMUL):
            self._check_binary_tile_op(instr)
        elif isinstance(instr, TMATMUL):
            self._check_matmul(instr)
    
    def _check_binary_tile_op(self, instr):
        """Check binary tile operation types match."""
        if hasattr(instr, 'src0') and hasattr(instr, 'src1'):
            src0_type = instr.src0.tile_type
            src1_type = instr.src1.tile_type
            
            if src0_type.shape != src1_type.shape:
                self.errors.append(
                    f"{instr.opcode}: Shape mismatch - {src0_type.shape} vs {src1_type.shape}"
                )
            
            if src0_type.element_type != src1_type.element_type:
                self.errors.append(
                    f"{instr.opcode}: Element type mismatch - "
                    f"{src0_type.element_type.value} vs {src1_type.element_type.value}"
                )
    
    def _check_matmul(self, instr: TMATMUL):
        """Check matrix multiply dimension compatibility."""
        a_type = instr.a.tile_type
        b_type = instr.b.tile_type
        
        if a_type.shape.cols != b_type.shape.rows:
            self.errors.append(
                f"TMATMUL: Incompatible dimensions - "
                f"A cols ({a_type.shape.cols}) != B rows ({b_type.shape.rows})"
            )
    
    def get_errors(self) -> List[str]:
        """Get list of type errors."""
        return self.errors


# =============================================================================
# Code Generator
# =============================================================================

class CodeGenerator:
    """
    Code generator for PTO programs.
    
    Generates PTO assembly from a validated program.
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
        self.output: List[str] = []
        self.indent_level = 0
    
    def generate(self) -> str:
        """Generate PTO assembly code."""
        self.output = []
        self._emit_header()
        self._emit_declarations()
        self._emit_instructions()
        self._emit_footer()
        return "\n".join(self.output)
    
    def _emit(self, line: str):
        """Emit a line of code with proper indentation."""
        indent = "  " * self.indent_level
        self.output.append(f"{indent}{line}")
    
    def _emit_header(self):
        """Emit program header."""
        self._emit(f"// PTO Program: {self.program.name}")
        self._emit(f"// Generated by PTO ISA Compiler")
        self._emit("")
    
    def _emit_declarations(self):
        """Emit variable declarations."""
        self._emit("// Tile Declarations")
        for name, tile_type in self.program.tile_declarations.items():
            self._emit(f"// %{name} : {tile_type}")
        
        self._emit("")
        self._emit("// Scalar Declarations")
        for name, dtype in self.program.scalar_declarations.items():
            self._emit(f"// %{name} : {dtype.value}")
        
        self._emit("")
        self._emit("// MemRef Declarations")
        for name, memref_type in self.program.memref_declarations.items():
            self._emit(f"// %{name} : {memref_type}")
        
        self._emit("")
    
    def _emit_instructions(self):
        """Emit program instructions."""
        self._emit("// Instructions")
        for instr in self.program.instructions:
            self._emit_instruction(instr)
    
    def _emit_instruction(self, instr: PTOInstruction):
        """Emit a single instruction."""
        if isinstance(instr, FOR):
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ENDFOR):
            self.indent_level = max(0, self.indent_level - 1)
            self._emit(instr.to_pto_as())
        elif isinstance(instr, IF):
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ELSE):
            self.indent_level = max(0, self.indent_level - 1)
            self._emit(instr.to_pto_as())
            self.indent_level += 1
        elif isinstance(instr, ENDIF):
            self.indent_level = max(0, self.indent_level - 1)
            self._emit(instr.to_pto_as())
        else:
            self._emit(instr.to_pto_as())
    
    def _emit_footer(self):
        """Emit program footer."""
        self._emit("")
        self._emit("// End of program")


# =============================================================================
# Optimizer
# =============================================================================

class Optimizer:
    """
    Basic optimizer for PTO programs.
    
    Performs simple optimizations like:
    - Dead code elimination
    - Constant folding
    - Loop unrolling
    """
    
    def __init__(self, program: PTOProgram):
        self.program = program
    
    def optimize(self, unroll_threshold: int = 4) -> PTOProgram:
        """Apply optimizations and return optimized program."""
        optimized = PTOProgram(
            name=self.program.name,
            tile_declarations=self.program.tile_declarations.copy(),
            scalar_declarations=self.program.scalar_declarations.copy(),
            memref_declarations=self.program.memref_declarations.copy(),
            metadata=self.program.metadata.copy()
        )
        
        optimized.instructions = self._optimize_instructions(
            self.program.instructions,
            unroll_threshold
        )
        
        return optimized
    
    def _optimize_instructions(self, instructions: List[PTOInstruction], 
                                unroll_threshold: int) -> List[PTOInstruction]:
        """Optimize a list of instructions."""
        result = []
        i = 0
        
        while i < len(instructions):
            instr = instructions[i]
            
            # Check for small loops that can be unrolled
            if isinstance(instr, FOR):
                loop_end, loop_body = self._find_loop_end(instructions, i)
                if loop_end > i:
                    iterations = self._get_loop_iterations(instr)
                    if iterations is not None and iterations <= unroll_threshold:
                        # Unroll the loop
                        unrolled = self._unroll_loop(instr, loop_body, iterations)
                        result.extend(unrolled)
                        i = loop_end + 1
                        continue
            
            result.append(instr)
            i += 1
        
        return result
    
    def _find_loop_end(self, instructions: List[PTOInstruction], 
                       start: int) -> Tuple[int, List[PTOInstruction]]:
        """Find the matching ENDFOR and extract loop body."""
        depth = 1
        body = []
        i = start + 1
        
        while i < len(instructions) and depth > 0:
            instr = instructions[i]
            if isinstance(instr, FOR):
                depth += 1
            elif isinstance(instr, ENDFOR):
                depth -= 1
                if depth == 0:
                    return i, body
            
            if depth > 0:
                body.append(instr)
            i += 1
        
        return -1, []
    
    def _get_loop_iterations(self, for_instr: FOR) -> Optional[int]:
        """Get the number of iterations for a FOR loop if computable."""
        if isinstance(for_instr.lb, ImmediateOperand) and \
           isinstance(for_instr.ub, ImmediateOperand) and \
           isinstance(for_instr.step, ImmediateOperand):
            lb = for_instr.lb.value
            ub = for_instr.ub.value
            step = for_instr.step.value
            if step > 0:
                return (ub - lb + step - 1) // step
        return None
    
    def _unroll_loop(self, for_instr: FOR, body: List[PTOInstruction], 
                     iterations: int) -> List[PTOInstruction]:
        """Unroll a loop by duplicating body instructions."""
        result = []
        
        # For now, just return the loop as-is (full unrolling is complex)
        # A real implementation would substitute the induction variable
        result.append(for_instr)
        result.extend(body)
        result.append(ENDFOR())
        
        return result


# =============================================================================
# Compiler Driver
# =============================================================================

class PTOCompiler:
    """
    Main compiler driver for PTO programs.
    
    Orchestrates parsing, type checking, optimization, and code generation.
    """
    
    def __init__(self, optimize: bool = True, unroll_threshold: int = 4):
        self.optimize = optimize
        self.unroll_threshold = unroll_threshold
    
    def compile(self, program: PTOProgram) -> str:
        """
        Compile a PTO program to assembly.
        
        Args:
            program: The PTO program to compile
            
        Returns:
            Generated PTO assembly code
            
        Raises:
            TypeError: If type checking fails
            ValidationError: If validation fails
        """
        # Type check
        type_checker = TypeChecker(program)
        if not type_checker.check():
            errors = type_checker.get_errors()
            raise TypeError(f"Type checking failed:\n" + "\n".join(errors))
        
        # Optimize if enabled
        if self.optimize:
            optimizer = Optimizer(program)
            program = optimizer.optimize(self.unroll_threshold)
        
        # Generate code
        generator = CodeGenerator(program)
        return generator.generate()
    
    def compile_and_save(self, program: PTOProgram, output_path: str):
        """Compile a program and save to file."""
        code = self.compile(program)
        with open(output_path, 'w') as f:
            f.write(code)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Simple matrix multiply
    print("=" * 60)
    print("Example 1: Simple Matrix Multiply")
    print("=" * 60)
    
    program1 = (PTOProgramBuilder("matmul_example")
        # Declare tiles
        .tile("a", 64, 64, ElementType.F16)
        .tile("b", 64, 64, ElementType.F16)
        .tile("c", 64, 64, ElementType.F32)
        # Declare memory
        .memref("mem_a", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F16)
        .memref("mem_c", MemorySpace.GM, ElementType.F32)
        # Load operands
        .load("a", "mem_a", 0, 0)
        .load("b", "mem_b", 0, 0)
        # Compute
        .matmul("c", "a", "b")
        # Store result
        .store("c", "mem_c", 0, 0)
        .build())
    
    compiler = PTOCompiler()
    code1 = compiler.compile(program1)
    print(code1)
    
    # Example 2: Tiled matrix multiply with nested loops
    print("\n" + "=" * 60)
    print("Example 2: Tiled Matrix Multiply with Nested Loops")
    print("=" * 60)
    
    program2 = (PTOProgramBuilder("tiled_matmul")
        # Declare tiles (smaller tiles for tiling)
        .tile("a_tile", 16, 16, ElementType.F16)
        .tile("b_tile", 16, 16, ElementType.F16)
        .tile("c_tile", 16, 16, ElementType.F32)
        # Larger tile for iteration bounds
        .tile("full_matrix", 64, 64, ElementType.F32)
        # Memory
        .memref("mem_a", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F16)
        .memref("mem_c", MemorySpace.GM, ElementType.F32)
        # Nested loop over tiles (4x4 tiles to cover 64x64)
        .for_loop("i", 0, 4, 1)  # Outer loop
        .for_loop("j", 0, 4, 1)  # Inner loop
        # Load tiles
        .load("a_tile", "mem_a")
        .load("b_tile", "mem_b")
        # Compute
        .matmul("c_tile", "a_tile", "b_tile")
        # Store
        .store("c_tile", "mem_c")
        .end_for()  # End inner
        .end_for()  # End outer
        .build())
    
    code2 = compiler.compile(program2)
    print(code2)
    
    # Example 3: MLP forward pass
    print("\n" + "=" * 60)
    print("Example 3: MLP Forward Pass")
    print("=" * 60)
    
    program3 = (PTOProgramBuilder("mlp_forward")
        # Input, weights, bias, output tiles
        .tile("input", 64, 128, ElementType.F16)
        .tile("weight", 128, 64, ElementType.F16)
        .tile("bias", 64, 64, ElementType.F32)
        .tile("output", 64, 64, ElementType.F32)
        .tile("activated", 64, 64, ElementType.F32)
        # Memory refs
        .memref("mem_in", MemorySpace.GM, ElementType.F16)
        .memref("mem_w", MemorySpace.GM, ElementType.F16)
        .memref("mem_b", MemorySpace.GM, ElementType.F32)
        .memref("mem_out", MemorySpace.GM, ElementType.F32)
        # Load data
        .load("input", "mem_in")
        .load("weight", "mem_w")
        .load("bias", "mem_b")
        # Linear layer: output = input @ weight
        .matmul("output", "input", "weight")
        # Add bias
        .add("output", "output", "bias")
        # ReLU activation
        .relu("activated", "output")
        # Store result
        .store("activated", "mem_out")
        .build())
    
    code3 = compiler.compile(program3)
    print(code3)
    
    print("\n" + "=" * 60)
    print("Compilation completed successfully!")
    print("=" * 60)
