# Runtime Kernel Compilation - Usage Guide

## Overview

The runtime kernel compilation feature allows you to compile AICore kernel source files (`.cpp`) at runtime using the `ccec` compiler, instead of compiling them at build time via CMake.

## Implementation Summary

### New Files Created
1. **[runtime/host/kernel_compiler.h](runtime/host/kernel_compiler.h)** - Runtime compiler interface
2. **[runtime/host/kernel_compiler.cpp](runtime/host/kernel_compiler.cpp)** - Compiler implementation
3. **[runtime/test_compiler.cpp](runtime/test_compiler.cpp)** - Standalone test for compilation

### Modified Files
1. **[runtime/host/devicerunner.h](runtime/host/devicerunner.h)** - Added `CompileAndLoadKernel()` and `LoadSingleKernelToDevice()`
2. **[runtime/host/devicerunner.cpp](runtime/host/devicerunner.cpp)** - Implementation + removed hardcoded kernel registration
3. **[runtime/CMakeLists.txt](runtime/CMakeLists.txt)** - Linked kernel_compiler.cpp
4. **[runtime/graphbuilder.cpp](runtime/graphbuilder.cpp)** - Updated to use runtime compilation

## How to Use

### Basic Usage

```cpp
#include "host/devicerunner.h"

// 1. Initialize DeviceRunner
DeviceRunner& runner = DeviceRunner::Get();
runner.Init(deviceId, numCores, "./aicpu/lib.so", "./aicore/kernel.o");

// 2. Compile and load kernels at runtime
std::string ptoIsaRoot = "/path/to/pto-isa";  // Or from PTO_ISA_ROOT env var
runner.CompileAndLoadKernel(0, "./aicore/kernels/kernel_add.cpp", ptoIsaRoot);
runner.CompileAndLoadKernel(1, "./aicore/kernels/kernel_mul.cpp", ptoIsaRoot);

// 3. Use the kernels in your graph
Graph graph;
graph.add_task(args, 4, 0);  // Uses func_id=0 (kernel_add)
runner.Run(graph);
```

### Environment Setup

Set the `PTO_ISA_ROOT` environment variable to point to the pto-isa headers:

```bash
export PTO_ISA_ROOT=/path/to/pto-isa-liao/runtime/build/_deps/pto-isa-src
```

If not set, the code will use a default path (`./build/_deps/pto-isa-src`).

### Standalone Compilation Test

```bash
cd runtime/build
export PTO_ISA_ROOT=$PWD/_deps/pto-isa-src
./test_compiler
```

Expected output:
```
=== Testing Runtime Kernel Compilation ===
Compiling: ../aicore/kernels/kernel_mul.cpp
Compilation successful: /tmp/kernel_<timestamp>_<pid>.o
✓ SUCCESS: Kernel compiled successfully
```

## API Reference

### KernelCompiler Class

```cpp
class KernelCompiler {
public:
    /**
     * Compile a kernel source file to an object file
     *
     * @param sourcePath  Path to kernel source file (.cpp)
     * @param ptoIsaRoot  Path to PTO-ISA root directory (headers location)
     * @param outputPath  [OUT] Path to compiled .o file (in /tmp)
     * @param errorMsg    [OUT] Error message if compilation fails
     * @return 0 on success, -1 on error
     */
    static int CompileKernel(const std::string& sourcePath,
                            const std::string& ptoIsaRoot,
                            std::string& outputPath,
                            std::string& errorMsg);
};
```

### DeviceRunner Methods

```cpp
/**
 * Compile and load a kernel at runtime
 *
 * Combines compilation, registration, and loading into a single call.
 *
 * @param funcId      Function identifier for this kernel (0, 1, 2, ...)
 * @param sourcePath  Path to kernel source file (.cpp)
 * @param ptoIsaRoot  Path to PTO-ISA root directory
 * @return 0 on success, -1 on error
 */
int CompileAndLoadKernel(int funcId,
                        const std::string& sourcePath,
                        const std::string& ptoIsaRoot);

/**
 * Load a single pre-compiled kernel binary
 *
 * Use this if you already have a compiled .o file.
 *
 * @param funcId   Function identifier
 * @param binPath  Path to compiled .o file
 * @return 0 on success, -1 on error
 */
int LoadSingleKernelToDevice(int funcId, const std::string& binPath);
```

## Compilation Details

### Compiler Flags Used

The runtime compiler uses the same flags as CMake build:
- `-c -O3 -g -x cce`
- `--cce-aicore-only`
- `--cce-aicore-arch=dav-c220-cube` (for AIC architecture)
- `-D__AIC__`
- Stack size: `-mllvm -cce-aicore-stack-size=0x8000`
- Function stack: `-mllvm -cce-aicore-function-stack-size=0x8000`
- Include paths: `-I${PTO_ISA_ROOT}/include` and `-I${PTO_ISA_ROOT}/include/pto`

### Output Location

Compiled `.o` files are stored in `/tmp` with unique names:
- Format: `/tmp/kernel_<timestamp>_<pid>.o`
- No caching - always recompiles from source
- Files can be cleaned up manually or on system reboot

### Requirements

1. **ASCEND_HOME_PATH** environment variable must be set
2. **ccec compiler** must be available at `${ASCEND_HOME_PATH}/bin/ccec`
3. **PTO-ISA headers** must be available (fetched at build time via CMake FetchContent)
4. **DeviceRunner** must be initialized before calling `CompileAndLoadKernel()`

## Example: Runtime Compilation in graphbuilder.cpp

```cpp
// Initialize device
DeviceRunner& runner = DeviceRunner::Get();
runner.Init(deviceId, 3, "./aicpu/libaicpu_graph_kernel.so", "./aicore/kernel.o");

// Get PTO-ISA root
const char* ptoIsaRootEnv = std::getenv("PTO_ISA_ROOT");
std::string ptoIsaRoot = ptoIsaRootEnv ? ptoIsaRootEnv : "./build/_deps/pto-isa-src";

// Compile and load all kernels
runner.CompileAndLoadKernel(0, "../aicore/kernels/kernel_add.cpp", ptoIsaRoot);
runner.CompileAndLoadKernel(1, "../aicore/kernels/kernel_add_scalar.cpp", ptoIsaRoot);
runner.CompileAndLoadKernel(2, "../aicore/kernels/kernel_mul.cpp", ptoIsaRoot);

// Build and run graph...
```

## Error Handling

All functions return 0 on success, -1 on error. Error messages are printed to stderr.

Common errors:
- **ASCEND_HOME_PATH not set**: Ensure environment variable is configured
- **ccec compiler not found**: Verify CANN toolkit installation
- **PTO-ISA headers not found**: Check PTO_ISA_ROOT path
- **Compilation failed**: Check compiler output in errorMsg parameter
- **Source file not found**: Verify sourcePath is correct

## Benefits

1. **No build-time compilation**: Kernels are not compiled during CMake build
2. **Dynamic loading**: Load kernels on-demand at runtime
3. **Flexible**: Compile different kernels based on runtime conditions
4. **Debugging**: Easier to test individual kernel changes without full rebuild
