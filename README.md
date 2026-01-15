# PTO ISA Compiler

A Domain-Specific Language (DSL) compiler for **Programmable Tensor Operations (PTO)** Instruction Set Architecture.

## Overview

The PTO ISA operates on **Tiles** - 2-dimensional blocks of data representing tensor slices. This compiler provides:

- **Complete ISA Definition**: All PTO instructions defined in Python
- **DSL for Program Construction**: Fluent interface for building PTO programs
- **Loop Constructs**: Single and nested loops with iteration counts derived from tile shapes
- **Loop Fusion Optimization**: Combines consecutive elementwise operations into single fused loops
- **ARM64 NEON Backend**: Generates optimized ARM64 NEON intrinsic code
- **Type Checking**: Validation of tile shapes and element types
- **Code Generation**: Output PTO assembly syntax or ARM64 C code

## Architecture

```
PTO_ISA_Compiler/
├── pto_isa_definition.py   # Complete PTO ISA instruction definitions
├── compiler.py             # Compiler infrastructure (parser, type checker, codegen)
├── pto_c_compiler.py       # PTO C to ARM64 C compiler
├── arm64_codegen.py        # ARM64 NEON intrinsic code generator
├── loop_fusion.py          # Loop fusion optimization pass
├── example_pto_sinh.c      # Example: sinh() using Taylor expansion
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Quick Start

### ARM64 Backend (C to C Compilation)

```bash
# Compile PTO C code to ARM64 NEON C code
python3 pto_c_compiler.py example_pto_sinh.c example_arm64_sinh.c

# Compile without loop fusion
python3 pto_c_compiler.py example_pto_sinh.c output.c --no-fusion
```

### Example: PTO C Syntax

```c
// Declare tiles
PTO_TILE(x, 8, 8, f32)
PTO_TILE(result, 8, 8, f32)

PTO_FUNCTION_START
void compute(float* input, float* output) {
    // Load from memory
    PTO_TLOAD(x, input, 0, 0)
    
    // Elementwise operations
    PTO_TMUL(result, x, x)         // result = x * x
    PTO_TADDS(result, result, 1.0f) // result = result + 1.0
    
    // Store to memory
    PTO_TSTORE(result, output, 0, 0)
}
PTO_FUNCTION_END
```

### Loop Fusion Optimization

The compiler automatically fuses consecutive elementwise operations of the same shape:

```
Before fusion (22 separate loops):
for (row) for (col) { result = x + x; }
for (row) for (col) { x_squared = x * x; }
for (row) for (col) { term = term * x_squared; }
...

After fusion (1 fused loop):
for (row) for (col) {
    result = x + x;
    x_squared = x * x;
    term = term * x_squared;
    ...
}
```

**Benefits:**
- Reduces loop overhead by ~95% (21 loop overheads saved for sinh example)
- Improves cache locality
- Code size reduction: 461 lines → 265 lines for sinh example

## Python DSL Usage

### Basic Usage

```python
from compiler import PTOProgramBuilder, PTOCompiler
from pto_isa_definition import ElementType, MemorySpace

# Build a simple matrix multiply program
program = (PTOProgramBuilder("matmul")
    # Declare tiles
    .tile("a", 64, 64, ElementType.F16)
    .tile("b", 64, 64, ElementType.F16)
    .tile("c", 64, 64, ElementType.F32)
    # Declare memory references
    .memref("mem_a", MemorySpace.GM, ElementType.F16)
    .memref("mem_b", MemorySpace.GM, ElementType.F16)
    .memref("mem_c", MemorySpace.GM, ElementType.F32)
    # Load operands
    .load("a", "mem_a", 0, 0)
    .load("b", "mem_b", 0, 0)
    # Matrix multiply
    .matmul("c", "a", "b")
    # Store result
    .store("c", "mem_c", 0, 0)
    .build())

# Compile to PTO assembly
compiler = PTOCompiler()
asm_code = compiler.compile(program)
print(asm_code)
```

### Loop Constructs

```python
# Single-level loop iterating over tile dimensions
program = (PTOProgramBuilder("loop_example")
    .tile("data", 64, 64, ElementType.F32)
    .memref("mem", MemorySpace.GM, ElementType.F32)
    # Loop based on tile rows
    .tile_loop("i", "data", "rows", step=1)
    .load("data", "mem")
    .end_for()
    .build())

# 2-level nested loop
program = (PTOProgramBuilder("nested_loop")
    .tile("data", 64, 64, ElementType.F32)
    .memref("mem", MemorySpace.GM, ElementType.F32)
    # Outer loop over rows, inner over columns
    .nested_tile_loop("i", "j", "data")
    .load("data", "mem")
    .end_nested_loop()
    .build())
```

## PTO ISA Instructions

### Tile Instructions

| Category | Instructions |
|----------|-------------|
| Memory | `TLOAD`, `TSTORE`, `MGATHER`, `MSCATTER` |
| Elementwise Unary | `TABS`, `TNEG`, `TNOT`, `TEXP`, `TLOG`, `TSQRT`, `TRSQRT`, `TRECIP`, `TRELU` |
| Elementwise Binary | `TADD`, `TSUB`, `TMUL`, `TDIV`, `TREM`, `TMAX`, `TMIN`, `TAND`, `TOR`, `TXOR`, `TSHL`, `TSHR` |
| Scalar Ops | `TADDS`, `TSUBS`, `TMULS`, `TDIVS`, `TMAXS`, `TMINS` |
| Matrix | `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`, `TMATMUL_MX` |
| Reduction | `TROWSUM`, `TROWMAX`, `TROWMIN`, `TCOLSUM`, `TCOLMAX`, `TCOLMIN` |
| Broadcast | `TEXPANDS`, `TROWEXPAND`, `TCOLEXPAND`, `TROWEXPANDMUL` |
| Data Movement | `TTRANS`, `TRESHAPE`, `TEXTRACT`, `TGATHER`, `TSCATTER` |

### ARM64 NEON Mapping

| PTO Instruction | ARM64 NEON Intrinsic |
|-----------------|---------------------|
| `TADD` | `vaddq_f32` |
| `TSUB` | `vsubq_f32` |
| `TMUL` | `vmulq_f32` |
| `TDIV` | `vdivq_f32` |
| `TABS` | `vabsq_f32` |
| `TNEG` | `vnegq_f32` |
| `TSQRT` | `vsqrtq_f32` |
| `TRELU` | `vmaxq_f32` (with zero) |
| `TADDS/TMULS/TDIVS` | Scalar broadcast + vector op |

### Scalar Instructions

| Category | Instructions |
|----------|-------------|
| Arithmetic | `ADD`, `SUB`, `MUL`, `DIV`, `REM`, `NEG`, `ABS` |
| Bitwise | `AND`, `OR`, `XOR`, `NOT`, `SHL`, `SHR`, `SAR` |
| Comparison | `CMP` |
| Memory | `LOAD`, `STORE` |
| Control | `MOV`, `LI`, `CVT` |

### Control Flow Instructions

| Category | Instructions |
|----------|-------------|
| Loops | `FOR`, `ENDFOR`, `WHILE`, `DO`, `ENDWHILE` |
| Conditional | `IF`, `ELSE`, `ENDIF` |
| Branch | `BR`, `BCOND`, `BREAK`, `CONTINUE` |
| Functions | `CALL`, `RET`, `YIELD` |

## Loop Fusion Details

### Fusion Rules

1. **Fusable Operations**: Elementwise operations (TADD, TSUB, TMUL, TDIV, TABS, TEXP, etc.)
2. **Same Shape Required**: Only operations on tiles with matching dimensions are fused
3. **Fusion Barriers**: Operations that stop fusion:
   - Reductions (TROWSUM, TCOLSUM)
   - Matrix operations (TMATMUL)
   - Control flow (FOR, ENDFOR)

### How It Works

```
Input PTO Instructions:
  TMUL(x_squared, x, x)      # Shape: 8x8
  TMULS(term, x, 1.0f)       # Shape: 8x8
  TMUL(term, term, x_squared) # Shape: 8x8
  TDIVS(term, term, 6.0f)    # Shape: 8x8
  TADD(result, result, term)  # Shape: 8x8

Fusion Analysis:
  All operations: same shape (8x8), all elementwise
  → Fuse into single loop

Output ARM64 Code:
  for (_row = 0; _row < 8; _row++) {
      for (_col = 0; _col < 8; _col += 4) {  // NEON: 4 floats
          // All 5 operations in one iteration
          x_squared[row][col] = x * x;
          term[row][col] = x * 1.0f;
          term[row][col] = term * x_squared;
          term[row][col] = term / 6.0f;
          result[row][col] = result + term;
      }
  }
```

## Data Types

### Element Types

```python
from pto_isa_definition import ElementType

# Floating point
ElementType.F16   # 16-bit float
ElementType.F32   # 32-bit float
ElementType.BF16  # Brain float 16

# Integer
ElementType.I8, ElementType.I16, ElementType.I32, ElementType.I64
ElementType.U8, ElementType.U16, ElementType.U32, ElementType.U64
```

### Tile Shapes

```python
from pto_isa_definition import TileShape, TileType

# Define a tile shape
shape = TileShape(rows=64, cols=64)

# Define a complete tile type
tile_type = TileType.create(rows=64, cols=64, dtype=ElementType.F32)
# -> !pto.tile<64x64xf32>
```

## Compiler Pipeline

```
                       ┌──────────────┐
PTO C Code ──────────> │   Parser     │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │ Loop Fusion  │  ← Combines elementwise ops
                       │  Optimizer   │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │   ARM64      │  ← Generates NEON intrinsics
                       │  Code Gen    │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │  ARM64 C     │
                       │   Output     │
                       └──────────────┘
```

## Running Examples

```bash
# Run the Python DSL compiler with built-in examples
python3 compiler.py

# Run the ISA definition examples
python3 pto_isa_definition.py

# Run loop fusion optimizer test
python3 loop_fusion.py

# Compile sinh example to ARM64
python3 pto_c_compiler.py example_pto_sinh.c example_arm64_sinh.c
```

## License

MIT License
