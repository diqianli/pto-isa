/**
 * PTO Runtime2 - Multi-Threaded Interface
 *
 * Extends the base runtime with multi-threaded execution support.
 * Orchestrator, Scheduler, and Workers run in separate threads.
 *
 * This header lives in ref_runtime/src/runtime/rt2/runtime/ (self-contained;
 * ref_runtime does not depend on pto-isa src/runtime2). Device orchestration
 * uses pto2_runtime_create_from_sm (non-threaded mode); this header is required
 * for compilation only; linking uses pto_runtime2_threaded_stub.c.
 */

#ifndef PTO_RUNTIME2_THREADED_H
#define PTO_RUNTIME2_THREADED_H

#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_worker.h"

// =============================================================================
// Threaded Runtime Structure
// =============================================================================

/**
 * Trace event for recording task execution
 */
typedef struct {
    int32_t task_id;
    int32_t worker_id;
    int64_t start_cycle;
    int64_t end_cycle;
    const char* func_name;
} PTO2TraceEvent;

/**
 * Maximum number of trace events
 */
#define PTO2_MAX_TRACE_EVENTS 65536

/**
 * Extended runtime with thread context
 */
typedef struct PTO2RuntimeThreaded {
    PTO2Runtime base;                 /* Base runtime (must be first) */
    PTO2ThreadContext thread_ctx;     /* Thread management */

    /* Contexts for threads */
    PTO2OrchestratorContext orch_ctx;
    PTO2SchedulerContext sched_ctx;

    /* Simulation mode flag */
    bool simulation_mode;

    /* Tracing */
    bool trace_enabled;
    const char* trace_filename;
    PTO2TraceEvent* trace_events;     /* Array of trace events */
    volatile int32_t trace_count;     /* Number of recorded events */
    pthread_mutex_t trace_mutex;      /* Mutex for thread-safe trace recording */

} PTO2RuntimeThreaded;

// =============================================================================
// Threaded Runtime Creation
// =============================================================================

PTO2RuntimeThreaded* pto2_runtime_create_threaded(int32_t num_cube_workers,
                                                  int32_t num_vector_workers,
                                                  bool simulation_mode);

PTO2RuntimeThreaded* pto2_runtime_create_threaded_custom(int32_t num_cube_workers,
                                                          int32_t num_vector_workers,
                                                          bool simulation_mode,
                                                          int32_t task_window_size,
                                                          int32_t heap_size,
                                                          int32_t dep_list_size);

void pto2_runtime_destroy_threaded(PTO2RuntimeThreaded* rt);
void pto2_runtime_reset_threaded(PTO2RuntimeThreaded* rt);

// =============================================================================
// Threaded Execution
// =============================================================================

typedef void (*PTO2OrchestrationFunc)(PTO2Runtime* rt, void* arg);

void pto2_runtime_run_threaded(PTO2RuntimeThreaded* rt,
                                PTO2OrchestrationFunc orchestration_func,
                                void* orchestration_arg);

void pto2_runtime_run_inline(PTO2RuntimeThreaded* rt,
                              PTO2OrchestrationFunc orchestration_func,
                              void* orchestration_arg);

// =============================================================================
// Thread Control
// =============================================================================

void pto2_runtime_start_threads(PTO2RuntimeThreaded* rt);
void pto2_runtime_stop_threads(PTO2RuntimeThreaded* rt);
void pto2_runtime_wait_completion(PTO2RuntimeThreaded* rt);
bool pto2_runtime_threads_done(PTO2RuntimeThreaded* rt);

// =============================================================================
// Tracing
// =============================================================================

void pto2_runtime_enable_trace(PTO2RuntimeThreaded* rt, const char* filename);
void pto2_runtime_record_trace(PTO2RuntimeThreaded* rt, int32_t task_id,
                                int32_t worker_id, int64_t start_cycle,
                                int64_t end_cycle, const char* func_name);
void pto2_runtime_write_trace(PTO2RuntimeThreaded* rt, const char* filename);

// =============================================================================
// Statistics
// =============================================================================

void pto2_runtime_print_threaded_stats(PTO2RuntimeThreaded* rt);
int64_t pto2_runtime_get_total_cycles(PTO2RuntimeThreaded* rt);

#endif /* PTO_RUNTIME2_THREADED_H */
