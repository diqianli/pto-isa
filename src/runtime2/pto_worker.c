/**
 * PTO Runtime2 - Worker Thread Implementation
 * 
 * Implements worker threads that execute tasks dispatched by the scheduler.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_worker.h"
#include "pto_runtime2.h"
#include "pto_runtime2_threaded.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// =============================================================================
// Worker Type Names
// =============================================================================

const char* pto2_worker_type_name(PTO2WorkerType type) {
    switch (type) {
        case PTO2_WORKER_CUBE:        return "CUBE";
        case PTO2_WORKER_VECTOR:      return "VECTOR";
        case PTO2_WORKER_AI_CPU:      return "AI_CPU";
        case PTO2_WORKER_ACCELERATOR: return "ACCELERATOR";
        default:                      return "UNKNOWN";
    }
}

// =============================================================================
// Cycle Estimation (for simulation mode)
// =============================================================================

// Default cycle costs
#define DEFAULT_CUBE_CYCLES      100   // Matrix multiply
#define DEFAULT_VECTOR_CYCLES    50    // Element-wise ops
#define DEFAULT_DMA_CYCLES       80    // DMA transfer
#define DEFAULT_AICPU_CYCLES     200   // AI_CPU operations

static int64_t estimate_cycles_by_name(const char* func_name, int64_t data_size) {
    if (!func_name) {
        return DEFAULT_VECTOR_CYCLES;
    }
    
    // Matrix operations (Cube)
    if (strstr(func_name, "gemm") || strstr(func_name, "matmul") ||
        strstr(func_name, "conv")) {
        return DEFAULT_CUBE_CYCLES + data_size / 1024;
    }
    
    // DMA / Copy operations
    if (strstr(func_name, "dma") || strstr(func_name, "copy") ||
        strstr(func_name, "transfer")) {
        return DEFAULT_DMA_CYCLES + data_size / 512;
    }
    
    // Vector operations
    if (strstr(func_name, "add") || strstr(func_name, "mul") ||
        strstr(func_name, "relu") || strstr(func_name, "sigmoid") ||
        strstr(func_name, "vector")) {
        return DEFAULT_VECTOR_CYCLES + data_size / 2048;
    }
    
    // Default: vector operation
    return DEFAULT_VECTOR_CYCLES + data_size / 2048;
}

// =============================================================================
// Worker Initialization
// =============================================================================

bool pto2_worker_init(PTO2WorkerContext* worker, int32_t worker_id,
                       PTO2WorkerType worker_type, struct PTO2Runtime* runtime) {
    memset(worker, 0, sizeof(PTO2WorkerContext));
    
    worker->worker_id = worker_id;
    worker->worker_type = worker_type;
    worker->runtime = runtime;
    worker->shutdown = false;
    worker->current_task_id = -1;
    
    return true;
}

void pto2_worker_destroy(PTO2WorkerContext* worker) {
    // Nothing to free for now
    (void)worker;
}

void pto2_worker_reset(PTO2WorkerContext* worker) {
    worker->tasks_executed = 0;
    worker->total_cycles = 0;
    worker->total_stall_cycles = 0;
    worker->current_task_id = -1;
}

// =============================================================================
// Completion Queue Implementation
// =============================================================================

bool pto2_completion_queue_init(PTO2CompletionQueue* queue, int32_t capacity) {
    queue->entries = (PTO2CompletionEntry*)calloc(capacity, sizeof(PTO2CompletionEntry));
    if (!queue->entries) {
        return false;
    }
    
    queue->capacity = capacity;
    queue->head = 0;
    queue->tail = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->entries);
        return false;
    }
    
    return true;
}

void pto2_completion_queue_destroy(PTO2CompletionQueue* queue) {
    if (queue->entries) {
        free(queue->entries);
        queue->entries = NULL;
    }
    pthread_mutex_destroy(&queue->mutex);
}

bool pto2_completion_queue_push(PTO2CompletionQueue* queue,
                                 int32_t task_id, int32_t worker_id,
                                 int64_t start_cycle, int64_t end_cycle) {
    pthread_mutex_lock(&queue->mutex);
    
    int32_t next_tail = (queue->tail + 1) % queue->capacity;
    if (next_tail == queue->head) {
        // Queue full
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    PTO2CompletionEntry* entry = &queue->entries[queue->tail];
    entry->task_id = task_id;
    entry->worker_id = worker_id;
    entry->start_cycle = start_cycle;
    entry->end_cycle = end_cycle;
    
    queue->tail = next_tail;
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool pto2_completion_queue_pop(PTO2CompletionQueue* queue, PTO2CompletionEntry* entry) {
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->head == queue->tail) {
        // Queue empty
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    *entry = queue->entries[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool pto2_completion_queue_empty(PTO2CompletionQueue* queue) {
    pthread_mutex_lock(&queue->mutex);
    bool empty = (queue->head == queue->tail);
    pthread_mutex_unlock(&queue->mutex);
    return empty;
}

// =============================================================================
// Task Acquisition and Execution
// =============================================================================

int32_t pto2_worker_get_task(PTO2WorkerContext* worker) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    // Get the ready queue for this worker type
    PTO2ReadyQueue* queue = &sched->ready_queues[worker->worker_type];
    pthread_mutex_t* mutex = &ctx->ready_mutex[worker->worker_type];
    pthread_cond_t* cond = &ctx->ready_cond[worker->worker_type];
    
    // Block until we get a task or shutdown
    return pto2_ready_queue_pop_threadsafe(queue, mutex, cond, &worker->shutdown);
}

int32_t pto2_worker_try_get_task(PTO2WorkerContext* worker) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    PTO2ReadyQueue* queue = &sched->ready_queues[worker->worker_type];
    pthread_mutex_t* mutex = &ctx->ready_mutex[worker->worker_type];
    
    return pto2_ready_queue_try_pop_threadsafe(queue, mutex);
}

void pto2_worker_execute_task(PTO2WorkerContext* worker, int32_t task_id) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2TaskDescriptor* task = pto2_sm_get_task(rt->base.sm_handle, task_id);
    
    worker->current_task_id = task_id;
    
    // Call the InCore function if provided
    if (task->func_ptr) {
        // Build args array from task outputs
        void* args[PTO2_MAX_OUTPUTS + PTO2_MAX_INPUTS];
        int num_args = 0;
        
        // Add output pointers first
        for (int i = 0; i < task->num_outputs; i++) {
            args[num_args++] = (char*)task->packed_buffer_base + task->output_offsets[i];
        }
        
        // Call the function
        PTO2InCoreFunc func = (PTO2InCoreFunc)task->func_ptr;
        func(args, num_args);
    }
    
    worker->current_task_id = -1;
    worker->tasks_executed++;
}

int64_t pto2_worker_simulate_task(PTO2WorkerContext* worker, int32_t task_id) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2TaskDescriptor* task = pto2_sm_get_task(rt->base.sm_handle, task_id);
    
    worker->current_task_id = task_id;
    
    // Calculate data size from packed buffer
    int64_t data_size = 0;
    if (task->packed_buffer_end && task->packed_buffer_base) {
        data_size = (int64_t)((char*)task->packed_buffer_end - 
                              (char*)task->packed_buffer_base);
    }
    
    // Estimate cycles
    int64_t cycles = estimate_cycles_by_name(task->func_name, data_size);
    
    worker->current_task_id = -1;
    worker->tasks_executed++;
    worker->total_cycles += cycles;
    
    return cycles;
}

void pto2_worker_task_complete(PTO2WorkerContext* worker, int32_t task_id, 
                                int64_t start_cycle, int64_t end_cycle) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Push completion to queue
    pto2_completion_queue_push(&ctx->completion_queue,
                                task_id, worker->worker_id,
                                start_cycle, end_cycle);
    
    // Signal completion condition
    pthread_mutex_lock(&ctx->done_mutex);
    pthread_cond_signal(&ctx->completion_cond);
    pthread_mutex_unlock(&ctx->done_mutex);
}

// =============================================================================
// Worker Thread Functions
// =============================================================================

void* pto2_worker_thread_func(void* arg) {
    PTO2WorkerContext* worker = (PTO2WorkerContext*)arg;
    
    while (!worker->shutdown) {
        // Get next task (blocks if queue empty)
        int32_t task_id = pto2_worker_get_task(worker);
        if (task_id < 0) {
            // Shutdown or error
            break;
        }
        
        // Execute the task
        pto2_worker_execute_task(worker, task_id);
        
        // Signal completion (with 0 cycles since not simulating)
        pto2_worker_task_complete(worker, task_id, 0, 0);
    }
    
    return NULL;
}

void* pto2_worker_thread_func_sim(void* arg) {
    PTO2WorkerContext* worker = (PTO2WorkerContext*)arg;
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    while (!worker->shutdown) {
        // Get next task (blocks if queue empty)
        int32_t task_id = pto2_worker_get_task(worker);
        if (task_id < 0) {
            // Shutdown or error
            break;
        }
        
        // Get task descriptor to check dependencies
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->base.sm_handle, task_id);
        
        // === Calculate earliest start based on dependencies ===
        int64_t earliest_start = 0;
        
        // Check all fanin tasks for their completion times
        pthread_mutex_lock(&ctx->task_end_mutex);
        int32_t fanin_current = task->fanin_head;
        while (fanin_current > 0) {
            PTO2DepListEntry* entry = pto2_dep_pool_get(&rt->base.orchestrator.dep_pool, 
                                                         fanin_current);
            if (!entry) break;
            
            int32_t dep_task_id = entry->task_id;
            int32_t dep_slot = PTO2_TASK_SLOT(dep_task_id);
            if (dep_slot >= 0 && dep_slot < ctx->task_end_cycles_capacity) {
                int64_t dep_end = ctx->task_end_cycles[dep_slot];
                if (dep_end > earliest_start) {
                    earliest_start = dep_end;
                }
            }
            
            fanin_current = entry->next_offset;
        }
        pthread_mutex_unlock(&ctx->task_end_mutex);
        
        // Get worker's current cycle (when this worker will be free)
        int64_t worker_free_cycle = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[worker->worker_id]);
        
        // Start time is max of dependency completion and worker availability
        int64_t start_cycle = (earliest_start > worker_free_cycle) ? 
                               earliest_start : worker_free_cycle;
        
        worker->task_start_cycle = start_cycle;
        
        // Simulate the task (estimate cycles)
        int64_t cycles = pto2_worker_simulate_task(worker, task_id);
        
        int64_t end_cycle = start_cycle + cycles;
        
        // Update task end cycle for dependency tracking
        int32_t slot = PTO2_TASK_SLOT(task_id);
        pthread_mutex_lock(&ctx->task_end_mutex);
        if (slot >= 0 && slot < ctx->task_end_cycles_capacity) {
            ctx->task_end_cycles[slot] = end_cycle;
        }
        pthread_mutex_unlock(&ctx->task_end_mutex);
        
        // Update worker's current cycle
        PTO2_STORE_RELEASE(&ctx->worker_current_cycle[worker->worker_id], end_cycle);
        
        // Track stall cycles
        int64_t stall_cycles = start_cycle - worker_free_cycle;
        if (stall_cycles > 0) {
            worker->total_stall_cycles += stall_cycles;
        }
        
        // Signal completion with actual timing
        pto2_worker_task_complete(worker, task_id, start_cycle, end_cycle);
    }
    
    return NULL;
}

// =============================================================================
// Statistics
// =============================================================================

void pto2_worker_print_stats(PTO2WorkerContext* worker) {
    printf("Worker %d (%s):\n", worker->worker_id, 
           pto2_worker_type_name(worker->worker_type));
    printf("  Tasks executed:     %lld\n", (long long)worker->tasks_executed);
    printf("  Total cycles:       %lld\n", (long long)worker->total_cycles);
    printf("  Total stall cycles: %lld\n", (long long)worker->total_stall_cycles);
    
    if (worker->tasks_executed > 0) {
        printf("  Avg cycles/task:    %lld\n", 
               (long long)(worker->total_cycles / worker->tasks_executed));
    }
}
