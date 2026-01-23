/**
 * PTO Runtime System - ARM64 Platform Header
 * 
 * ARM64-specific runtime components:
 * - Single ready queue management
 * - Distributed dependency tracking (part of task_complete)
 * - Multi-threaded worker execution
 * - Runtime entry point
 */

#ifndef PTO_RUNTIME_ARM64_H
#define PTO_RUNTIME_ARM64_H

#include "pto_runtime_common.h"

// =============================================================================
// ARM64 Platform Configuration
// =============================================================================

// ARM64 uses a single unified ready queue (no cube/vector separation)
// All workers pull from the same queue

// =============================================================================
// ARM64-Specific API
// =============================================================================

/**
 * ARM64 Runtime Entry Point - Multi-threaded task execution
 * 
 * This is the main entry point for executing PTO programs on ARM64.
 * 
 * Execution flow:
 * 1. Initialize runtime and spawn worker threads
 * 2. Call orchestration function to build task graph
 * 3. Workers execute tasks from ready queue in parallel
 * 4. Wait for all tasks to complete
 * 5. Shutdown workers and cleanup
 * 
 * @param orch_func               Orchestration function that builds the task graph
 * @param user_data               User data passed to orchestration function
 * @param num_workers             Number of worker threads (1-PTO_MAX_WORKERS)
 * @param execution_task_threshold  Task threshold to start execution:
 *                                  - 0: Wait until orchestration completes (default, safe)
 *                                  - >0: Start when active_task_count > threshold OR orchestration done
 *                                  This enables pipelining task graph building with execution.
 * @return 0 on success, -1 on failure
 */
int runtime_entry_arm64(PTOOrchFunc orch_func, void* user_data, int num_workers, 
                        int execution_task_threshold);

/**
 * Enable simulation mode for ARM64
 * In simulation mode, tasks call cycle_func instead of func_ptr
 * and execution timing is recorded to the global trace
 */
void pto_runtime_enable_simulation(PTORuntime* rt, int32_t num_workers);

// =============================================================================
// ARM64 Platform Implementation of Common API
// =============================================================================

// These functions are declared in pto_runtime_common.h but implemented
// in pto_runtime_arm64.c with ARM64-specific behavior:
//
// - pto_task_submit()           : Submits to single ready queue
// - pto_task_complete()         : Distributed dependency management
// - pto_task_complete_threadsafe(): Thread-safe version
// - pto_get_ready_task()        : Pop from single queue
// - pto_get_ready_task_blocking(): Blocking pop with condition variable
// - pto_loop_replay()           : Replay using single ready queue
// - pto_execute_all()           : Single-threaded execution
// - pto_execute_task_with_worker(): Execute task with trace recording

#endif // PTO_RUNTIME_ARM64_H
