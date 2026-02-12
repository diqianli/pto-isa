# PTO-ISA 架构概览 - 详细分析

## 概要

PTO-ISA (Parallel Tile Operation Instruction Set Architecture，并行分片操作指令集架构) 是华为昇腾NPU的虚拟指令集架构项目，提供了从Python DSL到AICore执行的完整分片计算开发栈。

**分析日期**: 2025-02-09
**项目位置**: `E:\cccode\pto-isa`
**已分析的关键文件**: 运行时、调度器和执行器模块中的47个源文件

---

## 1. 系统架构

### 1.1 四层架构模型

```
┌─────────────────────────────────────────────────────────────┐
│                   第1层：用户层 (User Layer)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Python DSL (pto/) 和 PTO-AS (pto_as/)              │  │
│  │  - PTO 类：内核构建的高级API                          │  │
│  │  - Tensor/Tile/Scalar 类型                           │  │
│  │  - PTO.build()：生成 PTO-AS 汇编文本                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ PTO-AS 文本
┌─────────────────────────────────────────────────────────────┐
│                  第2层：编译器层 (Compiler Layer)             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ptoas 编译器和 AST 前端                             │  │
│  │  - ptoas/python/ast_frontend.py：解析 Python         │  │
│  │  - 代码生成：生成 CCE C++ 内核代码                    │  │
│  │  - 注入统一 ABI：void kernel(__gm__ int64_t*)        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ C++ 内核 (.cpp)
┌─────────────────────────────────────────────────────────────┐
│                   第3层：运行时层 (Runtime Layer)             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Python 运行时 (pto_runtime.py)                      │  │
│  │  - Graph 类：任务图管理                              │  │
│  │  - DeviceRunner：设备初始化和执行                     │  │
│  │  - BinaryCompiler：编译和加载内核                     │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  C 运行时 (ref_runtime/)                             │  │
│  │  - Scheduler：依赖解析和任务分发                      │  │
│  │  - Shared Memory：任务描述符环形缓冲区                │  │
│  │  - Thread Management：调度器 + 工作线程              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ 二进制文件 (.so/.bin)
┌─────────────────────────────────────────────────────────────┐
│                第4层：硬件抽象层 (Hardware Abstraction)       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AICore 执行单元                                      │  │
│  │  - CUBE 单元：矩阵运算（矩阵乘法、卷积）              │  │
│  │  - VECTOR 单元：逐元素运算（加法、乘法）              │  │
│  │  - 统一内存：GM（全局内存）+ UB/L1 缓存              │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AICPU（控制处理器）                                  │  │
│  │  - Orchestrator 线程：构建任务依赖图                  │  │
│  │  - Scheduler 线程：解析依赖关系                       │  │
│  │  - Handshake 机制：向 AICore 分发任务                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 模块职责

#### 用户层 (`pto/`, `pto_as/`)
**位置**: `E:\cccode\pto-isa\pto\__init__.py`, `E:\cccode\pto-isa\pto_as\__init__.py`

**关键类**：
- `PTO`：内核构建的 Python DSL
  - `tensor()`：声明张量视图
  - `tile()`：声明分片操作
  - `scalar()`：声明标量值
  - `build()`：生成 PTO-AS 汇编文本

**使用示例**：
```python
from pto import PTO, scalar

pto = PTO("my_kernel")
A = pto.tensor(shape=(M, K), dtype="float32")
B = pto.tensor(shape=(K, N), dtype="float32")
C = pto.tensor(shape=(M, N), dtype="float32")
pto.tile_add(C, A, B)  # C = A + B
pto_as_text = pto.build()  # 生成 PTO-AS
```

#### 编译器层 (`src/compile/`, `ptoas/`)
**位置**: `E:\cccode\pto-isa\src\compile\pto_compile.py`

**关键函数**：
1. `PTOFunctionBuilder`：从 PTO-AS 构建 InCore 函数
2. `PTOModule`：多个内核的容器
3. `generate_ascend_code()`：生成 CCE C++ 内核代码
4. `ast_frontend`：解析 Python AST → PTO-AS → C++

**输出**：具有统一 ABI 签名的 CCE C++ 内核：
```cpp
extern "C" __aicore__ void kernel(__gm__ int64_t* args) {
    // args[0..n-1]：输入/输出指针
    // args[n]：元素数量
    // ... 内核实现 ...
}
```

#### 运行时层 (`pto_runtime.py`, `ref_runtime/`)
**位置**: `E:\cccode\pto-isa\pto_runtime.py`, `E:\cccode\pto-isa\ref_runtime\`

**关键组件**：

**Python 运行时** (`pto_runtime.py`):
- `Graph`：任务图构建
  - `add_task()`：添加任务及其参数
  - `add_successor()`：添加依赖边
- `DeviceRunner`：设备执行
  - `init()`：加载 AICPU/AICore 二进制文件
  - `run()`：在设备上执行任务图
  - `copyToDevice()`：将图传输到设备内存
- `BinaryCompiler`：将内核编译为二进制
  - `compile("aicore")`：ccec 编译
  - `compile("aicpu")`：gcc 编译
  - `compile("host")`：gcc 编译

**C 运行时** (`ref_runtime/src/runtime/rt2/`):
- **调度器** (`pto_scheduler.c/h`)：
  - 通过 fanin/fanout 引用计数进行依赖解析
  - 每个工作器类型的就绪队列（CUBE, VECTOR, AI_CPU, ACCELERATOR）
  - 流控的环形缓冲区管理
- **线程管理** (`pto_runtime2_threaded.h`)：
  - Orchestrator 线程：构建任务图
  - Scheduler 线程（3个）：解析依赖、分发任务
  - Worker 线程（AICore）：执行内核
- **共享内存** (`pto_shared_memory.h/c`)：
  - 任务描述符环形缓冲区
  - 依赖列表池
  - TensorMap 用于别名跟踪

#### 硬件抽象层 (`ref_runtime/src/runtime/rt2/aicore/`, `aicpu/`)
**位置**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicore\aicore_executor.cpp`
**位置**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicpu\aicpu_executor.cpp`

**AICore 执行器** (`aicore_executor.cpp`):
```cpp
__aicore__ void aicore_execute(__gm__ Runtime* runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake* my_hank = &runtime->workers[block_idx];

    // 阶段1：等待 AICPU 初始化
    while (my_hank->aicpu_ready == 0) { dcci(my_hank, ...); }

    // 阶段2：信号就绪
    my_hank->aicore_done = block_idx + 1;

    // 阶段3：主执行循环
    while (true) {
        dcci(my_hank, ...);  // 缓存失效

        if (my_hank->control == 1) break;  // 退出信号

        if (my_hank->task_status == 1 && my_hank->task != 0) {
            __gm__ PTO2DispatchPayload* payload = (__gm__ PTO2DispatchPayload*)my_hank->task;
            execute_task_from_payload(payload);  // 运行内核
            my_hank->task_status = 0;  // 标记完成
        }
    }
}
```

**AICPU 执行器** (`aicpu_executor.cpp`):
```cpp
int AicpuExecutor::resolve_and_dispatch_pto2(Runtime* runtime, ...) {
    // 一次性初始化：初始化 fanin_refcount 和就绪队列
    if (!pto2_init_done_.exchange(true)) {
        for (int32_t i = 0; i < task_count; i++) {
            PTO2TaskDescriptor* t = &task_descriptors[i & window_mask];
            if (t->fanin_count == 0) {
                // 无依赖：立即添加到就绪队列
                ready_queue_aic_[idx++] = i;
            }
        }
        pto2_init_complete_.store(true);
    } else {
        while (!pto2_init_complete_.load()) { std::this_thread::yield(); }
    }

    while (true) {
        // 阶段1：处理已完成的任务
        for (int i = 0; i < core_num; i++) {
            Handshake* h = &hank[core_id];
            if (h->task_status == 0 && h->task != 0) {
                PTO2DispatchPayload* payload = (PTO2DispatchPayload*)h->task;
                h->task = 0;

                // 更新消费者的 fanin_refcount
                int32_t fanout_head = pto2_task->fanout_head;
                while (fanout_head > 0) {
                    PTO2DepListEntry* entry = &dep_list_pool[fanout_head];
                    int32_t consumer_id = entry->task_id;
                    int prev = __atomic_fetch_add(&s_pto2_fanin_refcount[consumer_slot], 1);

                    // 如果所有依赖都满足，添加到就绪队列
                    if (prev + 1 == consumer_desc->fanin_count) {
                        ready_queue_aic_[idx++] = consumer_id;
                    }
                    fanout_head = entry->next_offset;
                }
                completed_tasks_++;
            }
        }

        // 阶段2：将就绪任务分发给空闲核心
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

## 2. 任务图执行模型

### 2.1 任务描述符结构

**位置**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_runtime2_types.h:305-339`

```c
typedef struct {
    // 标识
    int32_t task_id;              // 唯一任务ID（绝对值，非环绕）
    int32_t kernel_id;            // 要执行的 InCore 函数
    int32_t worker_type;          // CUBE(0) | VECTOR(1) | AI_CPU(2) | ACCELERATOR(3)
    int32_t scope_depth;          // 作用域嵌套深度

    // Fanin：此任务依赖的生产者（提交时设置一次）
    int32_t fanin_head;           // 第一个 fanin 条目的偏移量（0 = 空）
    int32_t fanin_count;          // 生产者依赖的数量

    // Fanout：依赖此任务的消费者（随消费者提交而增长）
    // 由 fanout_lock 保护（自旋锁）
    volatile int32_t fanout_lock;
    volatile int32_t fanout_head; // 第一个 fanout 条目的偏移量（0 = 空）
    volatile int32_t fanout_count;// 总消费者数 + scope_depth（用于生命周期）

    // 打包输出缓冲区（所有输出打包到单个连续缓冲区）
    void*    packed_buffer_base;
    void*    packed_buffer_end;
    int32_t  output_offsets[PTO2_MAX_OUTPUTS];  // 最多16个输出
    int32_t  num_outputs;

    // 输入缓冲区指针
    int32_t  num_inputs;

    // 函数指针
    void*    func_ptr;
    const char* func_name;

    // 状态标志
    bool     is_active;
} PTO2TaskDescriptor;
```

### 2.2 任务状态机

**位置**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_runtime2_types.h:90-96`

```
状态转换：
PENDING --[fanin_refcount == fanin_count]--> READY
READY --[分发到工作器]--> RUNNING
RUNNING --[内核返回]--> COMPLETED
COMPLETED --[fanout_refcount == fanout_count]--> CONSUMED
```

**状态定义**：
```c
typedef enum {
    PTO2_TASK_PENDING = 0,    // 等待依赖
    PTO2_TASK_READY = 1,      // 所有依赖满足，在就绪队列中
    PTO2_TASK_RUNNING = 2,    // 在工作器上执行
    PTO2_TASK_COMPLETED = 3,  // 执行完成，输出可能仍在使用
    PTO2_TASK_CONSUMED = 4    // 输出完全消费，缓冲区可释放
} PTO2TaskState;
```

**关键观察**：
1. **PENDING → READY**：当 `fanin_refcount == fanin_count`（所有生产者完成）时触发
2. **COMPLETED → CONSUMED**：当 `fanout_refcount == fanout_count`（所有消费者完成）时触发
3. **环形缓冲区回收**：状态 = CONSUMED 时任务槽可重用

### 2.3 依赖管理

**Fanin（生产者）**：
- **目的**：跟踪输入依赖（生产者数据被此任务消费）
- **结构**：通过 `fanin_head` → `PTO2DepListEntry[]` 的单向链表
- **生命周期**：在任务提交时设置一次，之后只读
- **引用计数**：`fanin_refcount` 随每个生产者完成递增

**Fanout（消费者）**：
- **目的**：跟踪哪些任务依赖此任务的输出
- **结构**：通过 `fanout_head` → `PTO2DepListEntry[]` 的单向链表
- **生命周期**：随消费者提交动态增长
- **保护**：`fanout_lock` 自旋锁同步协调器（添加消费者）和调度器（读取消费者）

**依赖列表条目**：
```c
typedef struct {
    int32_t task_id;          // 依赖/被依赖任务 ID
    int32_t next_offset;      // 下一个条目的偏移量（0 = 列表末尾）
} PTO2DepListEntry;
```

---

## 3. 调度架构

### 3.1 三层调度模型

```
┌──────────────────────────────────────────────────────────────┐
│  第1层：协调器 (当 thread_num=4 时的线程3)                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  职责：构建任务依赖图                                  │ │
│  │  - PTO2 提交：创建任务描述符                           │ │
│  │  - 初始化 fanin/fanout 列表                            │ │
│  │  - 设置初始任务状态（PENDING 或 READY）                │ │
│  │  - 调用 scope_end() 标记作用域边界                     │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                        → 将任务提交到共享内存
┌──────────────────────────────────────────────────────────────┐
│  第2层：调度器 (当 thread_num=4 时的线程0/1/2)               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  职责：解析依赖和分发                                  │ │
│  │  - 处理来自协调器的新任务                              │ │
│  │  - 任务完成时：更新 fanin_refcount                     │ │
│  │  - 检查消费者是否就绪：fanin_refcount == fanin_count   │ │
│  │  - 将就绪任务加入每个工作器类型的队列                  │ │
│  │  - 将任务分发给空闲的 AICore 工作器                    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                    → 通过 Handshake 分发
┌──────────────────────────────────────────────────────────────┐
│  第3层：工作器（AICore 线程）                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  职责：执行内核函数                                    │ │
│  │  - 轮询 Handshake 以获取任务分配                       │ │
│  │  - 解包 PTO2DispatchPayload                            │ │
│  │  - 调用 kernel(args)                                  │ │
│  │  - 将完成状态写入 Handshake                            │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 就绪队列架构

**位置**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.h:29-39`

```c
typedef struct {
    int32_t* task_ids;    // 任务 ID 的环形缓冲区
    int32_t  head;        // 出队位置
    int32_t  tail;        // 入队位置
    int32_t  capacity;    // 队列容量（65536 默认）
    int32_t  count;       // 当前任务数
} PTO2ReadyQueue;
```

**队列分布**：
- **每种工作器类型一个队列**：`ready_queues[PTO2_NUM_WORKER_TYPES]`
  - `ready_queues[PTO2_WORKER_CUBE]`：CUBE 任务
  - `ready_queues[PTO2_WORKER_VECTOR]`：VECTOR 任务
  - `ready_queues[PTO2_WORKER_AI_CPU]`：AI_CPU 任务
  - `ready_queues[PTO2_WORKER_ACCELERATOR]`：加速器任务

**设计原理**：
- **负载均衡**：独立队列防止工作器类型饥饿
- **公平性**：每个队列的 mutex/condvar 用于线程安全访问
- **效率**：环形缓冲区最小化内存分配

### 3.3 Handshake 机制

**位置**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\runtime.h:64-78`

```cpp
typedef struct {
    // 控制信号
    volatile uint32_t aicpu_ready;     // AICPU 初始化完成
    volatile uint32_t aicore_done;     // AICore 准备接收任务
    volatile uint32_t control;         // 1 = 退出信号

    // 任务分发
    volatile uint64_t task;            // PTO2DispatchPayload* 指针
    volatile uint32_t task_status;     // 0 = 空闲/完成，1 = 忙碌

    // 核心类型标识
    CoreType core_type;                // AIC 或 AIV
} Handshake;
```

**协议**：
1. **初始化阶段**：
   - AICPU 设置 `aicpu_ready = 1`
   - AICore 等待直到 `aicpu_ready == 1`
   - AICore 设置 `aicore_done = core_id + 1`
   - AICPU 等待直到所有 AICore 设置 `aicore_done`

2. **任务分发阶段**：
   - AICPU 构建包含内核参数的 `PTO2DispatchPayload`
   - AICPU 设置 `hank->task = (uint64_t)&payload`
   - AICPU 设置 `hank->task_status = 1`
   - AICore 在主循环中轮询 `task_status`

3. **任务执行阶段**：
   - AICore 从 `task` 读取 `PTO2DispatchPayload*`
   - AICore 调用 `kernel(args)`
   - AICore 完成时设置 `task_status = 0`

4. **关闭阶段**：
   - AICPU 在所有 handshake 缓冲区上设置 `control = 1`
   - AICore 在主循环中轮询 `control`，当 `control == 1` 时退出

---

## 4. 流控和内存管理

### 4.1 环形缓冲区指针

**位置**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.h:50-62`

```c
typedef struct PTO2SchedulerState {
    // 环形指针的本地副本（同步到共享内存）
    int32_t last_task_alive;      // 任务环形尾部（最旧的活动任务）
    int32_t heap_tail;            // 堆环形尾部（可回收内存）

    // 动态配置
    int32_t task_window_size;     // 任务窗口大小（2的幂，如16384）
    int32_t task_window_mask;     // task_window_size - 1（用于快速取模）

    // ...
} PTO2SchedulerState;
```

**指针语义**：
- `last_task_alive`：所有 `task_id < last_task_alive` 的任务都已 CONSUMED
- `heap_tail`：此偏移之前的内存可回收
- `current_task_index`：协调器提交的总任务数（在共享内存中）

**流控机制**：
1. 协调器在提交前检查 `current_task_index - last_task_alive < task_window_size`
2. 如果窗口满，协调器阻塞直到调度器推进 `last_task_alive`
3. 调度器在任务转换到 CONSUMED 时推进 `last_task_alive`

**槽位计算**：
```c
static inline int32_t pto2_task_slot(PTO2SchedulerState* sched, int32_t task_id) {
    return task_id & sched->task_window_mask;  // 快速取模（2的幂）
}
```

### 4.2 内存生命周期

**打包输出缓冲区设计**：
- **动机**：通过将所有输出打包到一个缓冲区来减少内存碎片
- **结构**：`packed_buffer_base` + `output_offsets[num_outputs]` + `packed_buffer_end`

```
任务描述符：
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
    GM 堆：[output0][output1][output2][...空闲...]
             ↑       ↑       ↑       ↑
             base   4096    8192    end
```

**内存回收**：
```c
void pto2_scheduler_advance_ring_pointers(PTO2SchedulerState* sched) {
    // 当任务为 CONSUMED 时推进 last_task_alive
    while (sched->last_task_alive < current_task_index) {
        int32_t slot = pto2_task_slot(sched, sched->last_task_alive);
        if (sched->task_state[slot] != PTO2_TASK_CONSUMED) {
            break;  // 发现未消费任务，停止推进
        }
        sched->last_task_alive++;
    }

    // 根据最后消费的任务的缓冲区更新 heap_tail
    if (sched->last_task_alive > 0) {
        int32_t last_consumed_id = sched->last_task_alive - 1;
        PTO2TaskDescriptor* last_consumed = pto2_sm_get_task(sched->sm_handle, last_consumed_id);
        if (last_consumed->packed_buffer_end != NULL) {
            sched->heap_tail = (int32_t)(intptr_t)last_consumed->packed_buffer_end;
        }
    }

    // 同步到共享内存以进行协调器流控
    pto2_scheduler_sync_to_sm(sched);
}
```

---

## 5. 关键文件位置

### 5.1 调度器核心
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.h` - 调度器接口（429行）
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.c` - 调度器实现（935行）

### 5.2 执行器
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicore\aicore_executor.cpp` - AICore 执行器（63行）
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\aicpu\aicpu_executor.cpp` - AICPU 调度器/执行器（1045行）

### 5.3 运行时
- `E:\cccode\pto-isa\pto_runtime.py` - Python 运行时入口（200+行）
- `E:\cccode\pto-isa\ref_runtime\python\binary_compiler.py` - 二进制编译器

### 5.4 类型定义
- `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_runtime2_types.h` - 核心类型定义（640行）

### 5.5 示例
- `E:\cccode\pto-isa\examples\bgemm\pto_bgemm.py` - BGEMM 内核定义
- `E:\cccode\pto-isa\examples\bgemm\run_ascend_a2a3.py` - BGEMM 执行脚本

---

## 6. 研究发现总结

### 6.1 架构优势
1. **关注点清晰分离**：具有清晰边界的4层架构
2. **高效调度**：每种工作器类型的就绪队列最小化锁竞争
3. **流控**：环形缓冲区窗口防止内存耗尽
4. **可移植设计**：支持多个平台（昇腾 A2/A3/A5、ARM64、CUDA）

### 6.2 关键设计决策
1. **打包输出缓冲区**：以复杂性为代价减少碎片
2. **环形缓冲区任务窗口**：通过2的幂窗口大小限制内存使用
3. **Fanout 锁**：同步协调器（添加消费者）和调度器（读取消费者）
4. **Handshake 轮询**：AICore 轮询任务（无中断）
5. **线程安全引用计数**：用于 fanin/fanout 跟踪的原子操作

### 6.3 性能特征
- **调度开销**：每个任务完成的依赖解析为 O(fanout)
- **锁竞争**：就绪队列的每个队列互斥锁（最小竞争）
- **内存开销**：环形缓冲区大小 = `task_window_size * sizeof(TaskDescriptor)`
- **可扩展性**：支持多达128个 AICore 工作器（可配置）

---

**下一步**：参见 `02_调度算法.md` 了解详细的算法分析。
