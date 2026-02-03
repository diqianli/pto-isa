/**
 * Stub implementations for PTO2 threaded runtime (tracing) when the full
 * pto_runtime2_threaded.c is not linked (e.g. ref_runtime AICPU device
 * orchestration uses pto2_runtime_create_from_sm; worker still calls
 * pto2_runtime_record_trace). No-op so trace is effectively disabled.
 */

#include "pto_runtime2_threaded.h"

void pto2_runtime_record_trace(PTO2RuntimeThreaded* rt, int32_t task_id,
                                int32_t worker_id, int64_t start_cycle,
                                int64_t end_cycle, const char* func_name) {
    (void)rt;
    (void)task_id;
    (void)worker_id;
    (void)start_cycle;
    (void)end_cycle;
    (void)func_name;
}
