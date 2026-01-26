/**
 * PTO Runtime - Ascend A2/A3 NPU Launcher Implementation
 * 
 * Implements NPU kernel launching using CANN SDK APIs.
 */

#include "a2a3_npu_launcher.h"
#include "a2a3_binary_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// CANN SDK Headers (when available)
// =============================================================================

#ifdef CANN_SDK_AVAILABLE
#include <acl/acl.h>
#include <runtime/rt.h>
#endif

// =============================================================================
// Internal Structures
// =============================================================================

#define MAX_INCORE_FUNCS 256

/**
 * InCore function entry in device GM.
 */
typedef struct {
    char func_name[64];
    uint64_t device_addr;     // Address in device GM
    size_t binary_size;
    bool is_cube;
    bool is_loaded;
} NPUInCoreEntry;

/**
 * NPU Launcher state.
 */
struct A2A3NPULauncher {
    A2A3NPULauncherConfig config;
    
    // Device state
    bool initialized;
    
#ifdef CANN_SDK_AVAILABLE
    // CANN resources
    aclrtStream stream_aicpu;
    aclrtStream stream_aicore;
    void* aicore_bin_handle;
#endif
    
    // InCore function registry
    NPUInCoreEntry incore_funcs[MAX_INCORE_FUNCS];
    int num_incore_funcs;
    
    // Handshake buffers
    PTOHandshake* host_handshake;
    void* device_handshake;
    
    // Kernel arguments
    PTOKernelArgs host_kargs;
    void* device_kargs;
};

// =============================================================================
// Helper Functions
// =============================================================================

static void log_verbose(A2A3NPULauncher* launcher, const char* fmt, ...) {
    if (launcher && launcher->config.verbose) {
        va_list args;
        va_start(args, fmt);
        printf("[NPU Launcher] ");
        vprintf(fmt, args);
        printf("\n");
        va_end(args);
    }
}

// =============================================================================
// NPU Launcher Implementation
// =============================================================================

A2A3NPULauncher* a2a3_npu_launcher_init(const A2A3NPULauncherConfig* config) {
    if (!config) {
        fprintf(stderr, "[NPU Launcher] Error: config is NULL\n");
        return NULL;
    }
    
    A2A3NPULauncher* launcher = (A2A3NPULauncher*)calloc(1, sizeof(A2A3NPULauncher));
    if (!launcher) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to allocate launcher\n");
        return NULL;
    }
    
    launcher->config = *config;
    launcher->num_incore_funcs = 0;
    
#ifdef CANN_SDK_AVAILABLE
    // Initialize ACL
    aclError rc = aclInit(NULL);
    if (rc != ACL_SUCCESS && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        fprintf(stderr, "[NPU Launcher] Error: aclInit failed: %d\n", rc);
        free(launcher);
        return NULL;
    }
    
    // Set device
    rc = aclrtSetDevice(config->device_id);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: aclrtSetDevice(%d) failed: %d\n",
                config->device_id, rc);
        free(launcher);
        return NULL;
    }
    
    // Create streams
    rc = aclrtCreateStream(&launcher->stream_aicpu);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to create AICPU stream: %d\n", rc);
        free(launcher);
        return NULL;
    }
    
    rc = aclrtCreateStream(&launcher->stream_aicore);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to create AICore stream: %d\n", rc);
        aclrtDestroyStream(launcher->stream_aicpu);
        free(launcher);
        return NULL;
    }
    
    // Load AICore kernel binary if provided
    if (config->aicore_kernel_path) {
        uint8_t* bin_data = NULL;
        size_t bin_size = 0;
        
        int load_rc = a2a3_load_elf_text_section(config->aicore_kernel_path, &bin_data, &bin_size);
        if (load_rc == 0 && bin_data && bin_size > 0) {
            rtDevBinary_t binary = {
                .magic = RT_DEV_BINARY_MAGIC_ELF,
                .version = 0,
                .data = bin_data,
                .length = bin_size
            };
            
            rc = rtRegisterAllKernel(&binary, &launcher->aicore_bin_handle);
            free(bin_data);
            
            if (rc != RT_ERROR_NONE) {
                fprintf(stderr, "[NPU Launcher] Error: rtRegisterAllKernel failed: %d\n", rc);
            } else {
                printf("[NPU Launcher] Loaded AICore kernel: %s (%zu bytes)\n",
                       config->aicore_kernel_path, bin_size);
            }
        }
    }
    
    // Allocate handshake buffers
    int total_cores = config->num_aic_cores + config->num_aiv_cores;
    size_t handshake_size = sizeof(PTOHandshake) * total_cores;
    
    launcher->host_handshake = (PTOHandshake*)calloc(total_cores, sizeof(PTOHandshake));
    if (!launcher->host_handshake) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to allocate host handshake\n");
        a2a3_npu_launcher_finalize(launcher);
        return NULL;
    }
    
    rc = aclrtMalloc(&launcher->device_handshake, handshake_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to allocate device handshake: %d\n", rc);
        a2a3_npu_launcher_finalize(launcher);
        return NULL;
    }
    
    // Initialize handshake buffers
    for (int i = 0; i < total_cores; i++) {
        launcher->host_handshake[i].aicpu_ready = 0;
        launcher->host_handshake[i].aicore_done = 0;
        launcher->host_handshake[i].control = 0;
        launcher->host_handshake[i].task = 0;
        launcher->host_handshake[i].task_status = 0;
        launcher->host_handshake[i].core_type = (i < config->num_aic_cores) ? 0 : 1;
    }
    
    printf("[NPU Launcher] Initialized: device=%d, cores=%d AIC + %d AIV\n",
           config->device_id, config->num_aic_cores, config->num_aiv_cores);
    
#else
    printf("[NPU Launcher] Warning: CANN SDK not available, using stub mode\n");
#endif
    
    launcher->initialized = true;
    
    // Load InCore functions from directories if specified
    if (config->incore_aic_dir) {
        a2a3_load_incore_binary_dir(config->incore_aic_dir, true);
    }
    if (config->incore_aiv_dir) {
        a2a3_load_incore_binary_dir(config->incore_aiv_dir, false);
    }
    
    return launcher;
}

uint64_t a2a3_npu_load_incore(A2A3NPULauncher* launcher,
                              const char* func_name,
                              const char* bin_path,
                              bool is_cube) {
    if (!launcher || !func_name || !bin_path) {
        return 0;
    }
    
    if (launcher->num_incore_funcs >= MAX_INCORE_FUNCS) {
        fprintf(stderr, "[NPU Launcher] Error: Max InCore functions reached\n");
        return 0;
    }
    
    // Load binary using ELF parser
    uint8_t* bin_data = NULL;
    size_t bin_size = 0;
    
    int rc = a2a3_load_elf_text_section(bin_path, &bin_data, &bin_size);
    if (rc != 0 || !bin_data || bin_size == 0) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to load %s\n", bin_path);
        return 0;
    }
    
    uint64_t device_addr = 0;
    
#ifdef CANN_SDK_AVAILABLE
    // Allocate device GM memory
    void* dev_ptr = NULL;
    aclError acl_rc = aclrtMalloc(&dev_ptr, bin_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to allocate device memory: %d\n", acl_rc);
        free(bin_data);
        return 0;
    }
    
    // Copy binary to device
    acl_rc = aclrtMemcpy(dev_ptr, bin_size, bin_data, bin_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to copy to device: %d\n", acl_rc);
        aclrtFree(dev_ptr);
        free(bin_data);
        return 0;
    }
    
    device_addr = (uint64_t)dev_ptr;
#else
    // Stub mode: use host address (for testing)
    device_addr = (uint64_t)bin_data;
    bin_data = NULL;  // Don't free - will be used as device address
#endif
    
    // Register in InCore function table
    NPUInCoreEntry* entry = &launcher->incore_funcs[launcher->num_incore_funcs];
    strncpy(entry->func_name, func_name, sizeof(entry->func_name) - 1);
    entry->device_addr = device_addr;
    entry->binary_size = bin_size;
    entry->is_cube = is_cube;
    entry->is_loaded = true;
    launcher->num_incore_funcs++;
    
    printf("[NPU Launcher] Loaded InCore: %s -> 0x%lx (%zu bytes) [%s]\n",
           func_name, device_addr, bin_size, is_cube ? "AIC" : "AIV");
    
    if (bin_data) free(bin_data);
    return device_addr;
}

uint64_t a2a3_npu_get_incore_addr(A2A3NPULauncher* launcher, const char* func_name) {
    if (!launcher || !func_name) return 0;
    
    for (int i = 0; i < launcher->num_incore_funcs; i++) {
        if (launcher->incore_funcs[i].is_loaded &&
            strcmp(launcher->incore_funcs[i].func_name, func_name) == 0) {
            return launcher->incore_funcs[i].device_addr;
        }
    }
    
    // Also check binary_loader registry
    A2A3InCoreBinaryEntry* entry = a2a3_lookup_incore_binary(func_name);
    if (entry && entry->device_addr != 0) {
        return entry->device_addr;
    }
    
    return 0;
}

int a2a3_npu_execute_graph(A2A3NPULauncher* launcher, PTOTaskGraph* graph) {
    if (!launcher || !launcher->initialized) {
        fprintf(stderr, "[NPU Launcher] Error: Launcher not initialized\n");
        return -1;
    }
    
    if (!graph || graph->num_tasks == 0) {
        return 0;
    }
    
    printf("[NPU Launcher] Executing graph with %d tasks\n", graph->num_tasks);
    
    // Set functionBinAddr for all tasks
    for (int i = 0; i < graph->num_tasks; i++) {
        PTOTask* task = &graph->tasks[i];
        uint64_t addr = a2a3_npu_get_incore_addr(launcher, task->func_name);
        task->functionBinAddr = addr;
        
        if (addr == 0) {
            fprintf(stderr, "[NPU Launcher] Warning: No binary for task %d (%s)\n",
                    i, task->func_name);
        }
    }
    
#ifdef CANN_SDK_AVAILABLE
    int total_cores = launcher->config.num_aic_cores + launcher->config.num_aiv_cores;
    size_t handshake_size = sizeof(PTOHandshake) * total_cores;
    
    // Copy handshake to device
    aclError rc = aclrtMemcpy(launcher->device_handshake, handshake_size,
                              launcher->host_handshake, handshake_size,
                              ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to copy handshake: %d\n", rc);
        return -1;
    }
    
    // Allocate and copy task graph to device
    size_t graph_size = sizeof(PTOTaskGraph);
    void* device_graph = NULL;
    rc = aclrtMalloc(&device_graph, graph_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to allocate graph: %d\n", rc);
        return -1;
    }
    
    rc = aclrtMemcpy(device_graph, graph_size, graph, graph_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: Failed to copy graph: %d\n", rc);
        aclrtFree(device_graph);
        return -1;
    }
    
    // Build kernel arguments
    launcher->host_kargs.hankArgs = (PTOHandshake*)launcher->device_handshake;
    launcher->host_kargs.graphArgs = (PTOTaskGraph*)device_graph;
    launcher->host_kargs.core_num = total_cores;
    launcher->host_kargs.aic_num = launcher->config.num_aic_cores;
    launcher->host_kargs.aiv_num = launcher->config.num_aiv_cores;
    
    // Launch AICPU kernel
    struct {
        PTOKernelArgs kargs;
        char kernel_name[32];
        char so_name[32];
        char op_name[32];
    } aicpu_args;
    
    aicpu_args.kargs = launcher->host_kargs;
    strcpy(aicpu_args.kernel_name, "DynTileFwkBackendKernelServer");
    strcpy(aicpu_args.so_name, "libaicpu_kernel.so");
    aicpu_args.op_name[0] = '\0';
    
    rtAicpuArgsEx_t rt_args = {
        .args = &aicpu_args,
        .argsSize = sizeof(aicpu_args),
        .kernelNameAddrOffset = offsetof(typeof(aicpu_args), kernel_name),
        .soNameAddrOffset = offsetof(typeof(aicpu_args), so_name)
    };
    
    rc = rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU",
                                       1, &rt_args, NULL, launcher->stream_aicpu, 0);
    if (rc != RT_ERROR_NONE) {
        fprintf(stderr, "[NPU Launcher] Error: AICPU launch failed: %d\n", rc);
        aclrtFree(device_graph);
        return -1;
    }
    
    // Launch AICore kernel
    if (launcher->aicore_bin_handle) {
        struct {
            PTOHandshake* hank_args;
        } aicore_args = {
            .hank_args = (PTOHandshake*)launcher->device_handshake
        };
        
        rtArgsEx_t aicore_rt_args = {
            .args = &aicore_args,
            .argsSize = sizeof(aicore_args)
        };
        
        rtTaskCfgInfo_t cfg = {
            .schemMode = RT_SCHEM_MODE_BATCH
        };
        
        rc = rtKernelLaunchWithHandleV2(launcher->aicore_bin_handle, 0, total_cores,
                                        &aicore_rt_args, NULL, launcher->stream_aicore, &cfg);
        if (rc != RT_ERROR_NONE) {
            fprintf(stderr, "[NPU Launcher] Error: AICore launch failed: %d\n", rc);
            aclrtFree(device_graph);
            return -1;
        }
    }
    
    // Synchronize
    rc = aclrtSynchronizeStream(launcher->stream_aicpu);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: AICPU sync failed: %d\n", rc);
    }
    
    rc = aclrtSynchronizeStream(launcher->stream_aicore);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: AICore sync failed: %d\n", rc);
    }
    
    // Copy back graph to get completion status
    rc = aclrtMemcpy(graph, graph_size, device_graph, graph_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(device_graph);
    
    printf("[NPU Launcher] Execution complete: %d tasks\n", graph->tasks_completed);
    return graph->tasks_completed;
    
#else
    // Stub mode: just count tasks as "executed"
    printf("[NPU Launcher] Stub mode: %d tasks (not actually executed)\n", graph->num_tasks);
    graph->tasks_completed = graph->num_tasks;
    return graph->num_tasks;
#endif
}

void* a2a3_npu_alloc(A2A3NPULauncher* launcher, size_t size) {
    if (!launcher || !launcher->initialized || size == 0) {
        return NULL;
    }
    
#ifdef CANN_SDK_AVAILABLE
    void* ptr = NULL;
    aclError rc = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "[NPU Launcher] Error: aclrtMalloc failed: %d\n", rc);
        return NULL;
    }
    return ptr;
#else
    return malloc(size);
#endif
}

void a2a3_npu_free(A2A3NPULauncher* launcher, void* ptr) {
    if (!launcher || !ptr) return;
    
#ifdef CANN_SDK_AVAILABLE
    aclrtFree(ptr);
#else
    free(ptr);
#endif
}

int a2a3_npu_copy_to_device(A2A3NPULauncher* launcher, void* dst, const void* src, size_t size) {
    if (!launcher || !dst || !src || size == 0) {
        return -1;
    }
    
#ifdef CANN_SDK_AVAILABLE
    aclError rc = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    return (rc == ACL_SUCCESS) ? 0 : -1;
#else
    memcpy(dst, src, size);
    return 0;
#endif
}

int a2a3_npu_copy_from_device(A2A3NPULauncher* launcher, void* dst, const void* src, size_t size) {
    if (!launcher || !dst || !src || size == 0) {
        return -1;
    }
    
#ifdef CANN_SDK_AVAILABLE
    aclError rc = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
    return (rc == ACL_SUCCESS) ? 0 : -1;
#else
    memcpy(dst, src, size);
    return 0;
#endif
}

void a2a3_npu_launcher_finalize(A2A3NPULauncher* launcher) {
    if (!launcher) return;
    
#ifdef CANN_SDK_AVAILABLE
    // Free InCore device memory
    for (int i = 0; i < launcher->num_incore_funcs; i++) {
        if (launcher->incore_funcs[i].is_loaded && launcher->incore_funcs[i].device_addr) {
            aclrtFree((void*)launcher->incore_funcs[i].device_addr);
        }
    }
    
    // Free handshake buffers
    if (launcher->device_handshake) {
        aclrtFree(launcher->device_handshake);
    }
    
    // Destroy streams
    if (launcher->stream_aicpu) {
        aclrtDestroyStream(launcher->stream_aicpu);
    }
    if (launcher->stream_aicore) {
        aclrtDestroyStream(launcher->stream_aicore);
    }
    
    // Finalize ACL
    aclFinalize();
#endif
    
    if (launcher->host_handshake) {
        free(launcher->host_handshake);
    }
    
    free(launcher);
    printf("[NPU Launcher] Finalized\n");
}
