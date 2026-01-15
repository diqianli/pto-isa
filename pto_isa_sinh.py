"""
PTO ISA Example: sinh() using Taylor Expansion (Python Version)

This file demonstrates the PTO Python DSL for computing sinh(x) on tiles.

Taylor expansion for sinh(x):
    sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
            = x + x³/6 + x⁵/120 + x⁷/5040 + ...

Algorithm:
    result = x
    term = x
    x_squared = x * x
    for n = 1 to N:
        term = term * x_squared / ((2n)(2n+1))
        result = result + term
"""

from compiler import PTOProgramBuilder, PTOCompiler
from pto_isa_definition import ElementType, MemorySpace
from loop_fusion import LoopFusionOptimizer, FusedLoop, FusedCodeGenerator, TileShape


# =============================================================================
# ARM64 Code Generator for Python DSL
# =============================================================================

def generate_arm64_from_program(program, enable_fusion=True):
    """
    Generate ARM64 NEON code from a PTO program with loop fusion.
    
    Args:
        program: PTOProgram built using PTOProgramBuilder
        enable_fusion: Whether to enable loop fusion optimization
        
    Returns:
        ARM64 C code as string
    """
    from arm64_codegen import ARM64CodeGen, ARM64_TYPE_MAP, generate_arm64_header
    from loop_fusion import (
        LoopFusionOptimizer, FusedLoop, FusionBarrier, FusedCodeGenerator,
        OpCategory, get_category, TileShape as FusionTileShape
    )
    from dataclasses import dataclass
    
    # Convert PTOProgram instructions to format suitable for fusion
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
    
    # Build tile info from program's tile_declarations
    tile_info = {}
    for name, tile_type in program.tile_declarations.items():
        tile_info[name] = MockTileInfo(
            name=name,
            rows=tile_type.shape.rows,
            cols=tile_type.shape.cols,
            dtype=tile_type.element_type.value
        )
    
    # Convert PTO instructions to mock instructions
    mock_instructions = []
    for instr in program.instructions:
        opcode = instr.__class__.__name__
        
        # Map instruction to mock format
        if opcode in ("TADD", "TSUB", "TMUL", "TDIV"):
            mock_instructions.append(MockInstruction(
                opcode=opcode,
                dst=instr.dst.name,
                operands=[instr.src0.name, instr.src1.name]
            ))
        elif opcode in ("TADDS", "TMULS", "TDIVS"):
            # scalar.name contains the string representation of the value
            scalar_str = instr.scalar.name
            if not scalar_str.endswith("f"):
                scalar_str += "f"
            mock_instructions.append(MockInstruction(
                opcode=opcode,
                dst=instr.dst.name,
                operands=[instr.src.name, scalar_str]
            ))
        elif opcode in ("TEXP", "TLOG", "TSQRT", "TRELU", "TABS", "TNEG"):
            mock_instructions.append(MockInstruction(
                opcode=opcode,
                dst=instr.dst.name,
                operands=[instr.src.name]
            ))
        elif opcode == "TLOAD":
            mock_instructions.append(MockInstruction(
                opcode="TLOAD",
                dst=instr.dst.name,
                operands=[instr.src_mem.name, "0", "0"]
            ))
        elif opcode == "TSTORE":
            mock_instructions.append(MockInstruction(
                opcode="TSTORE",
                dst=instr.dst_mem.name,
                operands=[instr.src.name, "0", "0"]
            ))
    
    # Generate code
    lines = []
    lines.append(generate_arm64_header())
    
    # Declare tiles
    for name, info in tile_info.items():
        c_type = ARM64_TYPE_MAP.get(info.dtype, "float")
        lines.append(f"{c_type} {name}[{info.rows}][{info.cols}];")
    lines.append("")
    
    # Apply fusion and generate
    if enable_fusion:
        optimizer = LoopFusionOptimizer(tile_info)
        fused_result = optimizer.optimize(mock_instructions)
        
        lines.append(f"// Loop fusion: {optimizer.stats['fusion_savings']} loop overheads saved")
        lines.append("")
        
        codegen = ARM64CodeGen()
        fused_codegen = FusedCodeGenerator()
        
        for item in fused_result:
            if isinstance(item, FusedLoop):
                fused_lines = fused_codegen.generate_fused_loop(item)
                lines.extend(fused_lines)
                lines.append("")
            elif isinstance(item, FusionBarrier):
                # Generate non-fused code for barrier
                instr = item.raw_instr
                info = tile_info.get(instr.dst)
                rows = info.rows if info else 8
                cols = info.cols if info else 8
                dtype = info.dtype if info else "f32"
                
                if instr.opcode == "TLOAD":
                    gen_lines = codegen.gen_tload(instr.dst, instr.operands[0], 0, 0, rows, cols, dtype)
                elif instr.opcode == "TSTORE":
                    gen_lines = codegen.gen_tstore(instr.operands[0], instr.dst, 0, 0, rows, cols, dtype)
                else:
                    gen_lines = [f"// {instr.opcode}: Not implemented"]
                lines.extend(gen_lines)
                lines.append("")
    else:
        # Generate without fusion
        codegen = ARM64CodeGen()
        for instr in mock_instructions:
            info = tile_info.get(instr.dst)
            rows = info.rows if info else 8
            cols = info.cols if info else 8
            dtype = info.dtype if info else "f32"
            
            if instr.opcode == "TADD":
                gen_lines = codegen.gen_tadd(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
            elif instr.opcode == "TMUL":
                gen_lines = codegen.gen_tmul(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
            elif instr.opcode == "TMULS":
                gen_lines = codegen.gen_tmuls(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
            elif instr.opcode == "TDIVS":
                gen_lines = codegen.gen_tdivs(instr.dst, instr.operands[0], instr.operands[1], rows, cols, dtype)
            elif instr.opcode == "TLOAD":
                gen_lines = codegen.gen_tload(instr.dst, instr.operands[0], 0, 0, rows, cols, dtype)
            elif instr.opcode == "TSTORE":
                gen_lines = codegen.gen_tstore(instr.operands[0], instr.dst, 0, 0, rows, cols, dtype)
            else:
                gen_lines = [f"// {instr.opcode}: Not implemented"]
            lines.extend(gen_lines)
            lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Build sinh() PTO Program using Python DSL
# =============================================================================

def build_sinh_program():
    """
    Build the sinh() computation using PTO Python DSL.
    
    sinh(x) = x + x³/3! + x⁵/5! + x⁷/7! + ...
    
    Uses 7 terms of Taylor expansion for good accuracy.
    """
    program = (PTOProgramBuilder("sinh_taylor")
        # ====================================================================
        # Tile Declarations
        # ====================================================================
        .tile("x", 8, 8, ElementType.F32)           # Input tile
        .tile("x_squared", 8, 8, ElementType.F32)   # x²
        .tile("term", 8, 8, ElementType.F32)        # Current Taylor term
        .tile("result", 8, 8, ElementType.F32)      # Accumulated result
        
        # Memory references
        .memref("input", MemorySpace.GM, ElementType.F32)
        .memref("output", MemorySpace.GM, ElementType.F32)
        
        # ====================================================================
        # Load input
        # ====================================================================
        .load("x", "input", 0, 0)
        
        # ====================================================================
        # Initialize
        # ====================================================================
        # result = x (first term of Taylor series)
        .muls("result", "x", 1.0)          # result = x * 1.0 = x
        
        # x_squared = x * x
        .mul("x_squared", "x", "x")        # x_squared = x²
        
        # term = x (starting term)
        .muls("term", "x", 1.0)            # term = x
        
        # ====================================================================
        # Taylor Expansion Terms
        # ====================================================================
        
        # Term 2: x³/3! = x³/6
        .mul("term", "term", "x_squared")  # term = x * x² = x³
        .divs("term", "term", 6.0)         # term = x³/6
        .add("result", "result", "term")   # result = x + x³/6
        
        # Term 3: x⁵/5! = x⁵/120
        # term = (x³/6) * x² / 20 = x⁵/120
        .mul("term", "term", "x_squared")  # term = x⁵/6
        .divs("term", "term", 20.0)        # term = x⁵/120
        .add("result", "result", "term")   # result += x⁵/120
        
        # Term 4: x⁷/7! = x⁷/5040
        # term = (x⁵/120) * x² / 42 = x⁷/5040
        .mul("term", "term", "x_squared")  # term = x⁷/120
        .divs("term", "term", 42.0)        # term = x⁷/5040
        .add("result", "result", "term")   # result += x⁷/5040
        
        # Term 5: x⁹/9! = x⁹/362880
        # term = (x⁷/5040) * x² / 72 = x⁹/362880
        .mul("term", "term", "x_squared")
        .divs("term", "term", 72.0)
        .add("result", "result", "term")
        
        # Term 6: x¹¹/11!
        # term = term * x² / 110
        .mul("term", "term", "x_squared")
        .divs("term", "term", 110.0)
        .add("result", "result", "term")
        
        # Term 7: x¹³/13!
        # term = term * x² / 156
        .mul("term", "term", "x_squared")
        .divs("term", "term", 156.0)
        .add("result", "result", "term")
        
        # ====================================================================
        # Store result
        # ====================================================================
        .store("result", "output", 0, 0)
        
        .build())
    
    return program


# =============================================================================
# Main: Generate PTO-AS and ARM64 code
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PTO ISA sinh() - Python DSL Version")
    print("=" * 70)
    
    # Build the program
    program = build_sinh_program()
    
    # =========================================================================
    # Output 1: PTO-AS Assembly
    # =========================================================================
    print("\n" + "=" * 70)
    print("PTO-AS Assembly Output")
    print("=" * 70 + "\n")
    
    compiler = PTOCompiler()
    pto_asm = compiler.compile(program)
    print(pto_asm)
    
    # Save PTO-AS to file
    with open("sinh_pto_as.txt", "w") as f:
        f.write(pto_asm)
    print("\n[Saved to sinh_pto_as.txt]")
    
    # =========================================================================
    # Output 2: ARM64 NEON Code (with Loop Fusion)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ARM64 NEON Code (with Loop Fusion)")
    print("=" * 70 + "\n")
    
    arm64_code = generate_arm64_from_program(program, enable_fusion=True)
    print(arm64_code)
    
    # Save ARM64 code to file
    with open("sinh_arm64_fused.c", "w") as f:
        f.write(arm64_code)
    print("\n[Saved to sinh_arm64_fused.c]")
    
    # =========================================================================
    # Output 3: ARM64 NEON Code (without Loop Fusion) for comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("ARM64 NEON Code (without Loop Fusion)")
    print("=" * 70 + "\n")
    
    arm64_code_nofusion = generate_arm64_from_program(program, enable_fusion=False)
    
    # Just show line count comparison
    fused_lines = len(arm64_code.split('\n'))
    nofused_lines = len(arm64_code_nofusion.split('\n'))
    
    print(f"Lines with fusion:    {fused_lines}")
    print(f"Lines without fusion: {nofused_lines}")
    print(f"Reduction:            {nofused_lines - fused_lines} lines ({100*(nofused_lines-fused_lines)//nofused_lines}%)")
    
    # Save non-fused version
    with open("sinh_arm64_nofusion.c", "w") as f:
        f.write(arm64_code_nofusion)
    print("\n[Saved to sinh_arm64_nofusion.c]")
    
    print("\n" + "=" * 70)
    print("Compilation Complete!")
    print("=" * 70)
