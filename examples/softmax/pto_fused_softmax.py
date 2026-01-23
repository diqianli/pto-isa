"""
PTO Dynamic Softmax with Orchestration

This module implements softmax with dynamic tiling using:
1. InCore functions: Basic tile-level operations (rowmax, rowexpandsub, exp, rowsum, rowexpanddiv)
2. Orchestration function: dynamic_softmax which schedules InCore function calls

Softmax computation:
    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_i - max(x)))

The orchestration function handles:
- Loop over tiles for arbitrary input sizes
- Tail handling for non-aligned sizes
- Task dependency management via PTO runtime
"""

import os
import sys

# Add src directory to path
_example_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_example_dir))
_src_dir = os.path.join(_root_dir, 'src')
sys.path.insert(0, _src_dir)

from compile.pto_compile import PTOFunctionBuilder, PTOModule
from isa_definition.pto_isa_definition import ElementType, MemorySpace, CompareMode

# Default configuration
DEFAULT_DTYPE = ElementType.F32
DEFAULT_ROWS = 8
DEFAULT_COLS = 8


# =============================================================================
# InCore Functions: Tile-level Operations
# =============================================================================

def create_rowmax_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Find maximum value in each row of a tile.
    
    Input: [rows, cols] tensor
    Output: [rows, 1] tensor with row-wise max values
    """
    return (PTOFunctionBuilder("rowmax")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowmax("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowexpandsub_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Subtract row-wise values from each element.
    
    Input x: [rows, cols] tensor
    Input row_vals: [rows, 1] tensor  
    Output: [rows, cols] tensor where output[i,j] = x[i,j] - row_vals[i,0]
    """
    return (PTOFunctionBuilder("rowexpandsub")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpandsub("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


def create_exp_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Element-wise exponential.
    
    Input: [rows, cols] tensor
    Output: [rows, cols] tensor where output[i,j] = exp(input[i,j])
    """
    return (PTOFunctionBuilder("elem_exp")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .exp("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowsum_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Sum values in each row of a tile.
    
    Input: [rows, cols] tensor
    Output: [rows, 1] tensor with row-wise sums
    """
    return (PTOFunctionBuilder("rowsum")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("result", rows, 1, dtype)
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input", 0, 0)
        .rowsum("result", "x")
        .store("result", "output", 0, 0)
        .build())


def create_rowexpanddiv_func(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    InCore: Divide each element by row-wise values.
    
    Input x: [rows, cols] tensor
    Input row_vals: [rows, 1] tensor
    Output: [rows, cols] tensor where output[i,j] = x[i,j] / row_vals[i,0]
    """
    return (PTOFunctionBuilder("rowexpanddiv")
        .in_core()
        .tile("x", rows, cols, dtype)
        .tile("row_vals", rows, 1, dtype)
        .tile("result", rows, cols, dtype)
        .memref("input_x", MemorySpace.GM, dtype)
        .memref("input_row", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        .load("x", "input_x", 0, 0)
        .load("row_vals", "input_row", 0, 0)
        .rowexpanddiv("result", "x", "row_vals")
        .store("result", "output", 0, 0)
        .build())


# =============================================================================
# Orchestration Function: Dynamic Softmax with Tiling Loop
# =============================================================================

def create_dynamic_softmax_module(rows=DEFAULT_ROWS, cols=DEFAULT_COLS, dtype=DEFAULT_DTYPE):
    """
    Create a module with dynamic softmax as orchestration function.
    
    The module contains:
    - InCore functions: rowmax, rowexpandsub, elem_exp, rowsum, rowexpanddiv
    - Orchestration function: dynamic_softmax (calls InCore functions in a loop)
    
    The orchestration function handles arbitrary input sizes by:
    1. Processing full tiles in a loop
    2. Handling tail rows separately
    
    Returns:
        PTOModule with all component functions
    """
    module = PTOModule("dynamic_softmax_module")
    
    # Add InCore building block functions
    module.add_function(create_rowmax_func(rows, cols, dtype))
    module.add_function(create_rowexpandsub_func(rows, cols, dtype))
    module.add_function(create_exp_func(rows, cols, dtype))
    module.add_function(create_rowsum_func(rows, cols, dtype))
    module.add_function(create_rowexpanddiv_func(rows, cols, dtype))
    
    # Create orchestration function that calls InCore functions with dynamic tiling
    dynamic_softmax = (PTOFunctionBuilder("dynamic_softmax", module=module)
        .not_in_core()  # Orchestration function
        
        # Memory references for input/output
        .memref("input", MemorySpace.GM, dtype)
        .memref("output", MemorySpace.GM, dtype)
        
        # Temporary buffers for intermediate results
        .memref("temp_rowmax", MemorySpace.GM, dtype)
        .memref("temp_shifted", MemorySpace.GM, dtype)
        .memref("temp_exp", MemorySpace.GM, dtype)
        .memref("temp_rowsum", MemorySpace.GM, dtype)
        
        # Scalar loop control variables
        .scalar("total_rows", ElementType.I32)       # Total input rows
        .scalar("tile_rows", ElementType.I32)        # Rows per tile (constant)
        .scalar("num_full_tiles", ElementType.I32)   # Number of full tiles
        .scalar("tail_rows", ElementType.I32)        # Remaining rows
        .scalar("has_tail", ElementType.U1)          # Whether tail exists
        .scalar("zero", ElementType.I32)
        
        # Initialize
        .scalar_li("tile_rows", rows)
        .scalar_li("zero", 0)
        
        # ====================================================================
        # Main Loop: Process full tiles
        # ====================================================================
        .for_loop("tile_idx", 0, "num_full_tiles", 1)
            # Call InCore functions for each tile
            # Each call becomes a task in the PTO runtime
            
            # Step 1: Find row maximum
            .call("rowmax", {"input": "input", "output": "temp_rowmax"})
            
            # Step 2: Subtract row max (for numerical stability)
            .call("rowexpandsub", {
                "input_x": "input",
                "input_row": "temp_rowmax",
                "output": "temp_shifted"
            })
            
            # Step 3: Compute exponential
            .call("elem_exp", {"input": "temp_shifted", "output": "temp_exp"})
            
            # Step 4: Sum each row
            .call("rowsum", {"input": "temp_exp", "output": "temp_rowsum"})
            
            # Step 5: Normalize by row sum
            .call("rowexpanddiv", {
                "input_x": "temp_exp",
                "input_row": "temp_rowsum",
                "output": "output"
            })
        .end_for()
        
        # ====================================================================
        # Handle Tail: Process remaining rows (if any)
        # ====================================================================
        .scalar_cmp("has_tail", "tail_rows", "zero", CompareMode.GT)
        .if_then("has_tail")
            # Same sequence for tail tile
            .call("rowmax", {"input": "input", "output": "temp_rowmax"})
            .call("rowexpandsub", {
                "input_x": "input",
                "input_row": "temp_rowmax",
                "output": "temp_shifted"
            })
            .call("elem_exp", {"input": "temp_shifted", "output": "temp_exp"})
            .call("rowsum", {"input": "temp_exp", "output": "temp_rowsum"})
            .call("rowexpanddiv", {
                "input_x": "temp_exp",
                "input_row": "temp_rowsum",
                "output": "output"
            })
        .endif()
        
        .build())
    
    module.add_function(dynamic_softmax)
    module.set_entry("dynamic_softmax")
    
    return module


# =============================================================================
# Main: Module Creation Only
# =============================================================================
# NOTE: Code generation, compilation, and simulation are handled by run_xxx.py
# scripts generated by config_example.py. This main() only creates and describes
# the module for demonstration purposes.

def main():
    """Create and describe the dynamic softmax module."""
    print("=" * 70)
    print("PTO Dynamic Softmax - Orchestration with InCore Functions")
    print("=" * 70)
    
    # Create module
    module = create_dynamic_softmax_module()
    
    print(f"\nModule: {module.name}")
    print(f"Entry:  {module.entry_function}")
    print(f"\nFunctions ({len(module.get_function_names())}):")
    
    incore_funcs = []
    orch_funcs = []
    for name in module.get_function_names():
        func = module.get_function(name)
        if func.is_in_core:
            incore_funcs.append(name)
        else:
            orch_funcs.append(name)
    
    print(f"\n  InCore ({len(incore_funcs)}):")
    for name in incore_funcs:
        print(f"    - {name}")
    
    print(f"\n  Orchestration ({len(orch_funcs)}):")
    for name in orch_funcs:
        print(f"    - {name}")
    
    print("\n" + "=" * 70)
    print("To generate code, compile, and run:")
    print("  1. Run: python config_example.py")
    print("  2. Select 'softmax' example")
    print("  3. Choose target platform")
    print("  4. Generate run_<platform>.py")
    print("  5. Run: python run_<platform>.py")
    print("=" * 70)
    
    return module


if __name__ == "__main__":
    main()
