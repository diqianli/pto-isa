# PTO-ISA Architecture Overview - Detailed Analysis

## Executive Summary

PTO-ISA (Parallel Tile Operation Instruction Set Architecture) is a virtual instruction set architecture for Huawei Ascend NPU, providing a complete tile computing development stack from Python DSL to AICore execution.

**Analysis Date**: 2025-02-09
**Project Location**: `E:\cccode\pto-isa`
**Key Files Analyzed**: 47 source files across runtime, scheduler, and executor modules

---

## 1. System Architecture

### 1.1 Four-Layer Architecture Model

```
┌─────────────────────────────────────────────────────────────┐
│                   Layer 1: User Layer                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Python DSL (pto/) & PTO-AS (pto_as/)                │  │
│  │  - PTO class: High-level kernel building API         │  │
│  │  - Tensor/Tile/Scalar types                          │  │
│  │  - PTO.build(): Emit PTO-AS assembly text            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ PTO-AS text
┌─────────────────────────────────────────────────────────────┐
│                  Layer 2: Compiler Layer                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ptoas Compiler & AST Frontend                       │  │
│  │  - ptoas/python/ast_frontend.py: Parse Python        │  │
│  │  - Code generation: Emit CCE C++ kernel code         │  │
│  │  - Inject unified ABI: void kernel(__gm__ int64_t*)  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ C++ kernel (.cpp)
┌─────────────────────────────────────────────────────────────┐
│                   Layer 3: Runtime Layer                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Python Runtime (pto_runtime.py)                     │  │
│  │  - Graph class: Task graph management                │  │
│  │  - DeviceRunner: Device initialization & execution    │  │
│  │  - BinaryCompiler: Compile and load kernels          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  C Runtime (ref_runtime/)                            │  │
│  │  - Scheduler: Dependency resolution & task dispatch  │  │
│  │  - Shared Memory: Task descriptor ring buffer        │  │
│  │  - Thread Management: Scheduler + Workers            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ Binary (.so/.bin)
┌─────────────────────────────────────────────────────────────┐
│                Layer 4: Hardware Abstraction                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AICore Execution Units                              │  │
│  │  - CUBE Unit: Matrix operations (matmul, conv)       │  │
│  │  - VECTOR Unit: Element-wise operations (add, mul)   │  │
│  │  - Unified memory: GM (Global Memory) + UB/L1 Cache  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AICPU (Control Processor)                           │  │
│  │  - Orchestrator Thread: Build task dependency graph  │  │
│  │  - Scheduler Threads: Resolve dependencies           │  │
│  │  - Handshake Mechanism: Dispatch tasks to AICore     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Module Responsibilities

#### User Layer (`pto/`, `pto_as/`)
**Location**: `E:\cccode\pto-isa\pto\__init__.py`, `E:\cccode\pto-isa\pto_as\__init__.py`

**Key Classes**:
- `PTO`: Python DSL for kernel construction
  - `tensor()`: Declare tensor views
  - `tile()`: Declare tile operations
  - `scalar()`: Declare scalar values
  - `build()`: Generate PTO-AS assembly text

**Example Usage**:
```python
from pto import PTO, scalar

pto = PTO("my_kernel")
A = pto.tensor(shape=(M, K), dtype="float32")
B = pto.tensor(shape=(K, N), dtype="float32")
C = pto.tensor(shape=(M, N), dtype="float32")
pto.tile_add(C, A, B)  # C = A + B
pto_as_text = pto.build()  # Emit PTO-AS
```

#### Compiler Layer (`src/compile/`, `ptoas/`)
**Location**: `E:\cccode\pto-isa\src\compile\pto_compile.py`

**Key Functions**:
1. `PTOFunctionBuilder`: Build InCore functions from PTO-AS
2. `PTOModule`: Container for multiple kernels
3. `generate_ascend_code()`: Emit CCE C++ kernel code
4. `ast_frontend`: Parse Python AST → PTO-AS → C++

**Output**: CCE C++ kernel with unified ABI signature:
```cpp
extern "C" __aicore__ void kernel(__gm__ int64_t* args) {
    // args[0..n-1]: input/output pointers
    // args[n]: size in elements
    // ... kernel implementation ...
}
```

#### Runtime Layer (`pto_runtime.py`, `ref_runtime/`)
**Location**: `E:\cccode\pto-isa\pto_runtime.py`, `E:\cccode\pto-isa\ref_runtime\`

**Key Components**:

**Python Runtime** (`pto_runtime.py`):
- `Graph`: Task graph construction
  - `add_task()`: Add task with arguments
  - `add_successor()`: Add dependency edge
- `DeviceRunner`: Device execution
  - `init()`: Load AICPU/AICore binaries
  - `run()`: Execute task graph on device
  - `copyToDevice()`: Transfer graph to device memory
- `BinaryCompiler`: Compile kernels to binaries
  - `compile("aicore")`: ccec compilation
  - `compile("aicpu")`: gcc compilation
  - `compile("host")`: gcc compilation

**C Runtime** (`ref_runtime/src/runtime/rt2/`):
- **Scheduler** (`pto_scheduler.c/h`):
  - Dependency resolution via fanin/fanout refcounts
  - Per-worker-type ready queues (CUBE, VECTOR, AI_CPU, ACCELERATOR)
  - Ring buffer management for flow control
- **Thread Management** (`pto_runtime2_threaded.h`):
  - Orchestrator thread: Build task graph
  - Scheduler threads (3): Resolve dependencies, dispatch tasks
  - Worker threads (AICore): Execute kernels
- **Shared Memory** (`pto_shared_memory.h/c`):
  - Task descriptor ring buffer
  - Dependency list pool
  - TensorMap for alias tracking

#### Hardware Abstraction Layer (`ref_runtime/src/runtime/rt2/aicore/`, `aicpu/`)
**Location**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicore\aicore_executor.cpp`
**Location**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicpu\aicpu_executor.cpp`

**AICore Executor** (`aicore_executor.cpp`):
```cpp
__aicore__ void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = &runtime->workers[block_idx];

    // Phase 1: Wait for AICPU initialization
    while (my_hank->aicpu_ready == 0) { dcci(my_hank, ...); }

    // Phase 2: Signal ready
    my_hank->aicore_done = block_idx + 1;

    // Phase 3: Main execution loop
    while (true) {
        dcci(my_hank, ...);  // Cache invalidate

        if (my_hank->control == 1) break;  // Quit signal

        if (my_hank->task_status == 1 && my_hank->task != 0) {
            __gm__ PTO2DispatchPayload* payload = (__gm__ PTO2DispatchPayload*)my_hank->task;
            execute_task_from_payload(payload);  // Run kernel
            my_hank->task_status = 0;  // Mark complete
        }
    }
}
```

**AICPU Executor** (`aicpu_executor.cpp`):
```cpp
int AicpuExecutor::resolve_and_dispatch_pto2(Runtime* runtime, ...) {
    // One-time init: Initialize fanin_refcount and ready queue
    if (!pto2_init_done_.exchange(true)) {
        for (int32_t i = 0; i < task_count; i++) {
            PTO2TaskDescriptor* t = &task_descriptors[i & window_mask];
            if (t->fanin_count == 0) {
                // No dependencies: add to ready queue immediately
                ready_queue_aic_[idx++] = i;
            }
        }
        pto2_init_complete_.store(true);
    } else {
        while (!pto2_init_complete_.load()) { std::this_thread::yield(); }
    }

    while (true) {
        // Phase 1: Process completed tasks
        for (int i = 0; i < core_num; i++) {
            Handshake* h = &hank[core_id];
            if (h->task_status == 0 && h->task != 0) {
                PTO2DispatchPayload* payload = (PTO2DispatchPayload*)h->task;
                h->task = 0;

                // Update fanin_refcount of consumers
                int32_t fanout_head = pto2_task->fanout_head;
                while (fanout_head > 0) {
                    PTO2DepListEntry* entry = &dep_list_pool[fanout_head];
                    int32_t consumer_id = entry->task_id;
                    int prev = __atomic_fetch_add(&s_pto2_fanin_refcount[consumer_slot], 1);

                    // If all dependencies satisfied, add to ready queue
                    if (prev + 1 == consumer_desc->fanin_count) {
                        ready_queue_aic_[idx++] = consumer_id;
                    }
                    fanout_head = entry->next_offset;
                }
                completed_tasks_++;
            }
        }

        // Phase 2: Dispatch ready tasks to idle cores
        for (int i = 0; i < core_num; i++) {
            Handshake* h = &hank[core_id];
            if (h->task_status == 0 && h->task == 0) {
                if (ready_count_aic_.load() > 0) {
                    int32_t task_id = ready_queue_aic_[--count];
                    PTO2TaskDescriptor* task = &task_descriptors[task_id & window_mask];
                    build_pto2_payload(&payload, ...);
                    h->task = (uint64_t)&payload;
                    h->task_status = 1;
                }
            }
        }

        if (completed_tasks_ >= task_count && all_cores_idle) break;
    }
}
```

---

## 2. Task Graph Execution Model

### 2.1 Task Descriptor Structure

**Location**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_runtime2_types.h:305-339`

```c
typedef struct {
    // Identification
    int32_t task_id;              // Unique task ID (absolute, not wrapped)
    int32_t kernel_id;            // InCore function to execute
    int32_t worker_type;          // CUBE(0) | VECTOR(1) | AI_CPU(2) | ACCELERATOR(3)
    int32_t scope_depth;          // Scope nesting depth

    // Fanin: Producers this task depends on (set once at submission)
    int32_t fanin_head;           // Offset to first fanin entry (0 = empty)
    int32_t fanin_count;          // Number of producer dependencies

    // Fanout: Consumers that depend on this task (grows as consumers submit)
    // PROTECTED BY fanout_lock (spinlock)
    volatile int32_t fanout_lock;
    volatile int32_t fanout_head; // Offset to first fanout entry (0 = empty)
    volatile int32_t fanout_count;// Total consumers + scope_depth (for lifecycle)

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void*    packed_buffer_base;
    void*    packed_buffer_end;
    int32_t  output_offsets[PTO2_MAX_OUTPUTS];  // Up to 16 outputs
    int32_t  num_outputs;

    // Input buffer pointers
    int32_t  num_inputs;

    // Function pointer
    void*    func_ptr;
    const char* func_name;

    // Status flags
    bool     is_active;
} PTO2TaskDescriptor;
```

### 2.2 Task State Machine

**Location**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_runtime2_types.h:90-96`

```
State Transitions:
PENDING --[fanin_refcount == fanin_count]--> READY
READY --[dispatched to worker]--> RUNNING
RUNNING --[kernel returns]--> COMPLETED
COMPLETED --[fanout_refcount == fanout_count]--> CONSUMED
```

**State Definitions**:
```c
typedef enum {
    PTO2_TASK_PENDING = 0,    // Waiting for dependencies
    PTO2_TASK_READY = 1,      // All dependencies satisfied, in ready queue
    PTO2_TASK_RUNNING = 2,    // Currently executing on worker
    PTO2_TASK_COMPLETED = 3,  // Execution finished, output may be in use
    PTO2_TASK_CONSUMED = 4    // Output fully consumed, buffers releasable
} PTO2TaskState;
```

**Key Observations**:
1. **PENDING → READY**: Triggered when `fanin_refcount == fanin_count` (all producers completed)
2. **COMPLETED → CONSUMED**: Triggered when `fanout_refcount == fanout_count` (all consumers finished)
3. **Ring Buffer Recycling**: Task slot can be reused when state = CONSUMED

### 2.3 Dependency Management

**Fanin (Producers)**:
- **Purpose**: Track input dependencies (tasks that produce data this task consumes)
- **Structure**: Singly-linked list via `fanin_head` → `PTO2DepListEntry[]`
- **Lifecycle**: Set once at task submission, read-only after
- **Refcount**: `fanin_refcount` increments as each producer completes

**Fanout (Consumers)**:
- **Purpose**: Track which tasks depend on this task's output
- **Structure**: Singly-linked list via `fanout_head` → `PTO2DepListEntry[]`
- **Lifecycle**: Grows dynamically as consumers submit
- **Protection**: `fanout_lock` spinlock synchronizes orchestrator (adding consumers) and scheduler (reading consumers)

**Dependency List Entry**:
```c
typedef struct {
    int32_t task_id;          // Dependent/dependency task ID
    int32_t next_offset;      // Offset to next entry (0 = end of list)
} PTO2DepListEntry;
```

---

## 3. Scheduling Architecture

### 3.1 Three-Layer Scheduling Model

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: Orchestrator (Thread 3 when thread_num=4)         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Responsibility: Build task dependency graph           │ │
│  │  - PTO2 submission: Create task descriptors            │ │
│  │  - Initialize fanin/fanout lists                       │ │
│  │  - Set initial task state (PENDING or READY)           │ │
│  │  - Call scope_end() to mark scope boundaries           │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                        ↓ submit tasks to shared memory
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: Scheduler (Threads 0/1/2 when thread_num=4)       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Responsibility: Resolve dependencies & dispatch       │ │
│  │  - Process new tasks from orchestrator                 │ │
│  │  - On task completion: Update fanin_refcount           │ │
│  │  - Check if consumers ready: fanin_refcount == fanin_count │
│  │  - Enqueue ready tasks to per-worker-type queues       │ │
│  │  - Dispatch tasks to idle AICore workers               │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                    ↓ dispatch via Handshake
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: Workers (AICore Threads)                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Responsibility: Execute kernel functions              │ │
│  │  - Poll Handshake for task assignment                  │ │
│  │  - Unpack PTO2DispatchPayload                          │ │
│  │  - Call kernel(args)                                   │ │
│  │  - Write completion status to Handshake                │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Ready Queue Architecture

**Location**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.h:29-39`

```c
typedef struct {
    int32_t* task_ids;    // Circular buffer of task IDs
    int32_t  head;        // Dequeue position
    int32_t  tail;        // Enqueue position
    int32_t  capacity;    // Queue capacity (65536 default)
    int32_t  count;       // Current number of tasks
} PTO2ReadyQueue;
```

**Queue Distribution**:
- **One queue per worker type**: `ready_queues[PTO2_NUM_WORKER_TYPES]`
  - `ready_queues[PTO2_WORKER_CUBE]`: CUBE tasks
  - `ready_queues[PTO2_WORKER_VECTOR]`: VECTOR tasks
  - `ready_queues[PTO2_WORKER_AI_CPU]`: AI_CPU tasks
  - `ready_queues[PTO2_WORKER_ACCELERATOR]`: Accelerator tasks

**Design Rationale**:
- **Load Balancing**: Separate queues prevent worker type starvation
- **Fairness**: Per-queue mutex/condvar for thread-safe access
- **Efficiency**: Circular buffer minimizes memory allocation

### 3.3 Handshake Mechanism

**Location**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\runtime.h:64-78`

```cpp
typedef struct {
    // Control signals
    volatile uint32_t aicpu_ready;     // AICPU initialization complete
    volatile uint32_t aicore_done;     // AICore ready to receive tasks
    volatile uint32_t control;         // 1 = quit signal

    // Task dispatch
    volatile uint64_t task;            // PTO2DispatchPayload* pointer
    volatile uint32_t task_status;     // 0 = idle/completed, 1 = busy

    // Core type identification
    CoreType core_type;                // AIC or AIV
} Handshake;
```

**Protocol**:
1. **Initialization Phase**:
   - AICPU sets `aicpu_ready = 1`
   - AICore waits until `aicpu_ready == 1`
   - AICore sets `aicore_done = core_id + 1`
   - AICPU waits until all AICores set `aicore_done`

2. **Task Dispatch Phase**:
   - AICPU builds `PTO2DispatchPayload` with kernel args
   - AICPU sets `hank->task = (uint64_t)&payload`
   - AICPU sets `hank->task_status = 1`
   - AICore polls `task_status` in main loop

3. **Task Execution Phase**:
   - AICore reads `PTO2DispatchPayload*` from `task`
   - AICore calls `kernel(args)`
   - AICore sets `task_status = 0` on completion

4. **Shutdown Phase**:
   - AICPU sets `control = 1` on all handshake buffers
   - AICore polls `control` in main loop, exits when `control == 1`

---

## 4. Flow Control and Memory Management

### 4.1 Ring Buffer Pointers

**Location**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.h:50-62`

```c
typedef struct PTO2SchedulerState {
    // Local copies of ring pointers (synced to shared memory)
    int32_t last_task_alive;      // Task ring tail (oldest alive task)
    int32_t heap_tail;            // Heap ring tail (reclaimable memory)

    // Dynamic configuration
    int32_t task_window_size;     // Task window size (power of 2, e.g., 16384)
    int32_t task_window_mask;     // task_window_size - 1 (for fast modulo)

    // ...
} PTO2SchedulerState;
```

**Pointer Semantics**:
- `last_task_alive`: All tasks with `task_id < last_task_alive` are CONSUMED
- `heap_tail`: Memory up to this offset can be reclaimed
- `current_task_index`: Total tasks submitted by orchestrator (in shared memory)

**Flow Control Mechanism**:
1. Orchestrator checks `current_task_index - last_task_alive < task_window_size` before submitting
2. If window full, orchestrator blocks until scheduler advances `last_task_alive`
3. Scheduler advances `last_task_alive` when tasks transition to CONSUMED

**Slot Calculation**:
```c
static inline int32_t pto2_task_slot(PTO2SchedulerState* sched, int32_t task_id) {
    return task_id & sched->task_window_mask;  // Fast modulo (power of 2)
}
```

### 4.2 Memory Lifecycle

**Packed Output Buffer Design**:
- **Motivation**: Reduce memory fragmentation by packing all outputs into one buffer
- **Structure**: `packed_buffer_base` + `output_offsets[num_outputs]` + `packed_buffer_end`

```
Task Descriptor:
┌─────────────────────────────────────────────┐
│ packed_buffer_base ─────────────────────┐   │
│ output_offsets[0] = 0                    │   │
│ output_offsets[1] = 4096                 │   │
│ output_offsets[2] = 8192                 │   │
│ num_outputs = 3                          │   │
│ packed_buffer_end ───────────────────┐   │   │
└─────────────────────────────────────│───│---┘
                                      │   │
                                      ↓   ↓
    GM Heap: [output0][output1][output2][...free...]
             ↑       ↑       ↑       ↑
             base   4096    8192    end
```

**Memory Reclamation**:
```c
void pto2_scheduler_advance_ring_pointers(PTO2SchedulerState* sched) {
    // Advance last_task_alive while tasks are CONSUMED
    while (sched->last_task_alive < current_task_index) {
        int32_t slot = pto2_task_slot(sched, sched->last_task_alive);
        if (sched->task_state[slot] != PTO2_TASK_CONSUMED) {
            break;  // Found non-consumed task, stop advancing
        }
        sched->last_task_alive++;
    }

    // Update heap_tail based on last consumed task's buffer
    if (sched->last_task_alive > 0) {
        int32_t last_consumed_id = sched->last_task_alive - 1;
        PTO2TaskDescriptor* last_consumed = pto2_sm_get_task(sched->sm_handle, last_consumed_id);
        if (last_consumed->packed_buffer_end != NULL) {
            sched->heap_tail = (int32_t)(intptr_t)last_consumed->packed_buffer_end;
        }
    }

    // Sync to shared memory for orchestrator flow control
    pto2_scheduler_sync_to_sm(sched);
}
```

---

## 5. Key File Locations

### 5.1 Scheduler Core
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.h` - Scheduler interface (429 lines)
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.c` - Scheduler implementation (935 lines)

### 5.2 Executors
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicore\aicore_executor.cpp` - AICore executor (63 lines)
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicpu\aicpu_executor.cpp` - AICPU scheduler/executor (1045 lines)

### 5.3 Runtime
- `E:\cccode\pto-isa\pto_runtime.py` - Python runtime entry (200+ lines)
- `E:\cccode\pto-isa\ref_runtime\python\binary_compiler.py` - Binary compiler

### 5.4 Types
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_runtime2_types.h` - Core type definitions (640 lines)

### 5.5 Examples
- `E:\cccode\pto-isa\examples\bgemm\pto_bgemm.py` - BGEMM kernel definition
- `E:\cccode\pto-isa\examples\bgemm\run_ascend_a2a3.py` - BGEMM execution script

---

## 6. Summary of Findings

### 6.1 Architecture Strengths
1. **Clean Separation of Concerns**: 4-layer architecture with clear boundaries
2. **Efficient Scheduling**: Per-worker-type ready queues minimize lock contention
3. **Flow Control**: Ring buffer window prevents memory exhaustion
4. **Portable Design**: Supports multiple platforms (Ascend A2/A3/A5, ARM64, CUDA)

### 6.2 Key Design Decisions
1. **Packed Output Buffers**: Reduce fragmentation at cost of complexity
2. **Ring Buffer Task Window**: Limit memory usage with power-of-2 size
3. **Fanout Lock**: Synchronize orchestrator (adding consumers) and scheduler (reading consumers)
4. **Handshake Polling**: AICore polls for tasks (no interrupts)
5. **Thread-Safe Refcounts**: Atomic operations for fanin/fanout tracking

### 6.3 Performance Characteristics
- **Scheduling Overhead**: O(fanout) per task completion for dependency resolution
- **Lock Contention**: Per-queue mutex for ready queues (minimal contention)
- **Memory Overhead**: Ring buffer size = `task_window_size * sizeof(TaskDescriptor)`
- **Scalability**: Supports up to 128 AICore workers (configurable)

---

**Next**: See `02_scheduling_algorithms.md` for detailed algorithm analysis.
