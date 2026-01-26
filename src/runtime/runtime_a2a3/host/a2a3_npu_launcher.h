/**
 * PTO Runtime - Ascend A2/A3 NPU Launcher
 * 
 * This module handles launching kernels on the Ascend NPU:
 * - Loading AICore kernel binary to device
 * - Loading InCore function binaries to device GM
 * - Starting AICPU kernel via CANN API
 * - Starting AICore kernel via CANN API
 * - Building and copying task graph to device
 * 
 * Requires CANN SDK for actual hardware execution.
 */

#ifndef A2A3_NPU_LAUNCHER_H
#define A2A3_NPU_LAUNCHER_H

#include "../core/common/pto_task.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// NPU Launcher Configuration
// =============================================================================

/**
 * NPU launcher configuration.
 */
typedef struct {
    // Device configuration
    int device_id;                    // Ascend device ID
    
    // Kernel binaries
    const char* aicore_kernel_path;   // Path to aicore_kernel.o
    const char* aicpu_kernel_path;    // Path to libaicpu_kernel.so (optional)
    
    // InCore function directories
    const char* incore_aic_dir;       // Directory with AIC .o files
    const char* incore_aiv_dir;       // Directory with AIV .o files
    
    // Core configuration
    int32_t num_aic_cores;            // Number of AIC (Cube) cores
    int32_t num_aiv_cores;            // Number of AIV (Vector) cores
    
    // Debug options
    bool verbose;                     // Enable verbose logging
} A2A3NPULauncherConfig;

/**
 * Default NPU launcher configuration.
 */
#define A2A3_NPU_LAUNCHER_CONFIG_DEFAULT { \
    .device_id = 0, \
    .aicore_kernel_path = NULL, \
    .aicpu_kernel_path = NULL, \
    .incore_aic_dir = NULL, \
    .incore_aiv_dir = NULL, \
    .num_aic_cores = 24, \
    .num_aiv_cores = 48, \
    .verbose = false \
}

// =============================================================================
// NPU Launcher State
// =============================================================================

/**
 * NPU launcher state - opaque handle.
 */
typedef struct A2A3NPULauncher A2A3NPULauncher;

// =============================================================================
// NPU Launcher API
// =============================================================================

/**
 * Initialize NPU launcher.
 * 
 * This function:
 * 1. Sets the device
 * 2. Loads AICore kernel binary
 * 3. Loads InCore function binaries to device GM
 * 4. Allocates handshake buffers
 * 
 * @param config Launcher configuration
 * @return Launcher handle, or NULL on failure
 */
A2A3NPULauncher* a2a3_npu_launcher_init(const A2A3NPULauncherConfig* config);

/**
 * Load InCore function binary to device GM.
 * 
 * @param launcher Launcher handle
 * @param func_name Function name
 * @param bin_path Path to .o file
 * @param is_cube True if AIC (Cube) function
 * @return Device GM address, or 0 on failure
 */
uint64_t a2a3_npu_load_incore(A2A3NPULauncher* launcher,
                              const char* func_name,
                              const char* bin_path,
                              bool is_cube);

/**
 * Get InCore function device address by name.
 * 
 * @param launcher Launcher handle
 * @param func_name Function name
 * @return Device GM address, or 0 if not found
 */
uint64_t a2a3_npu_get_incore_addr(A2A3NPULauncher* launcher, const char* func_name);

/**
 * Execute task graph on NPU.
 * 
 * This function:
 * 1. Copies task graph to device
 * 2. Launches AICPU kernel
 * 3. Launches AICore kernel
 * 4. Waits for completion
 * 
 * @param launcher Launcher handle
 * @param graph Task graph to execute
 * @return Number of tasks executed, or negative on failure
 */
int a2a3_npu_execute_graph(A2A3NPULauncher* launcher, PTOTaskGraph* graph);

/**
 * Allocate device memory.
 * 
 * @param launcher Launcher handle
 * @param size Size in bytes
 * @return Device pointer, or NULL on failure
 */
void* a2a3_npu_alloc(A2A3NPULauncher* launcher, size_t size);

/**
 * Free device memory.
 * 
 * @param launcher Launcher handle
 * @param ptr Device pointer
 */
void a2a3_npu_free(A2A3NPULauncher* launcher, void* ptr);

/**
 * Copy data to device.
 * 
 * @param launcher Launcher handle
 * @param dst Device pointer
 * @param src Host pointer
 * @param size Size in bytes
 * @return 0 on success, negative on failure
 */
int a2a3_npu_copy_to_device(A2A3NPULauncher* launcher, void* dst, const void* src, size_t size);

/**
 * Copy data from device.
 * 
 * @param launcher Launcher handle
 * @param dst Host pointer
 * @param src Device pointer
 * @param size Size in bytes
 * @return 0 on success, negative on failure
 */
int a2a3_npu_copy_from_device(A2A3NPULauncher* launcher, void* dst, const void* src, size_t size);

/**
 * Finalize NPU launcher.
 * 
 * Releases all device resources.
 * 
 * @param launcher Launcher handle
 */
void a2a3_npu_launcher_finalize(A2A3NPULauncher* launcher);

#ifdef __cplusplus
}
#endif

#endif // A2A3_NPU_LAUNCHER_H
