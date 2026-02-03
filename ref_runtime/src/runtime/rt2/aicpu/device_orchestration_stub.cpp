/**
 * Device orchestration stub - weak symbol for AICPU thread 3
 *
 * When use_device_orchestration is true, AICPU thread 3 calls
 * aicpu_orchestration_entry(sm_ptr, args, arg_count). This stub provides a
 * default (empty) implementation: it only sets orchestrator_done in the shared
 * memory header so that scheduler threads can proceed. Users can override this
 * symbol by linking their own implementation that builds the task graph on
 * device.
 *
 * Shared memory header layout (PTO2SharedMemoryHeader): orchestrator_done is
 * at offset 8 (after current_task_index and heap_top).
 */
#include <cstdint>

#define PTO2_ORCHESTRATOR_DONE_OFFSET 8

extern "C" {

__attribute__((weak)) void aicpu_orchestration_entry(void* sm_ptr,
                                                     uint64_t* args,
                                                     int arg_count) {
    (void)args;
    (void)arg_count;
    if (sm_ptr != nullptr) {
        *reinterpret_cast<volatile int32_t*>(
            static_cast<char*>(sm_ptr) + PTO2_ORCHESTRATOR_DONE_OFFSET) = 1;
    }
}

}  // extern "C"
