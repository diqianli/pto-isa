# PTO-ISA 调度算法 - 详细分析

## 概要

本文档提供了PTO-ISA核心调度算法的全面分析，包括依赖解析、任务分发、流控和内存管理。所有代码引用均来自实际实现。

**分析日期**: 2025-02-09
**主要来源**: `E:\cccode\pto-isa\ref_runtime\src\runtime\rt2\runtime\pto_scheduler.c`（935行）

---

## 1. 依赖解析算法

### 1.1 算法概述

**位置**: `pto_scheduler.c:420-466` - `pto2_scheduler_on_task_complete()`

**目的**：当任务完成时，更新所有消费者的 `fanin_refcount`，并在所有依赖满足时将它们转换到READY状态。

**伪代码**：
```
算法: OnTaskComplete(task_id)
输入: completed_task_id
输出: 新就绪的任务已入队

1. slot ← task_id & task_window_mask
2. task ← GetTaskDescriptor(task_id)
3. task_state[slot] ← COMPLETED
4. tasks_completed++

5. // === 步骤1：更新消费者的 fanin_refcount ===
6. Acquire(task.fanout_lock)  // 防止协调器添加消费者
7. current ← task.fanout_head
8. while current > 0:
9.     entry ← DepListPool[current]
10.    consumer_id ← entry.task_id
11.    consumer_slot ← consumer_id & task_window_mask
12.    consumer ← GetTaskDescriptor(consumer_id)
13.
14.    fanin_refcount[consumer_slot]++
15.
16.    if fanin_refcount[consumer_slot] == consumer.fanin_count:
17.        task_state[consumer_slot] ← READY
18.        Enqueue(consumer_id, ready_queues[consumer.worker_type])
19.        WakeUpWorkers(consumer.worker_type)
20.
21.    current ← entry.next_offset
22. Release(task.fanout_lock)

23. // === 步骤2：更新生产者的 fanout_refcount ===
24. current ← task.fanin_head
25. while current > 0:
26.    entry ← DepListPool[current]
27.    producer_id ← entry.task_id
28.    ReleaseProducer(producer_id)
29.    current ← entry.next_offset

30. // === 步骤3：检查此任务是否可以转换到 CONSUMED ===
31. CheckAndHandleConsumed(task_id, task)
```

### 1.2 线程安全版本（多线程模式）

**位置**: `pto_scheduler.c:637-730` - `on_task_complete_threadsafe()`

**关键区别**：
1. **Fanout 锁**：在读取 `fanout_list` 之前获取自旋锁（防止与协调器添加消费者竞争）
2. **原子引用计数更新**：使用 `__atomic_add_fetch()` 进行 `fanin_refcount`
3. **CAS 状态转换**：使用 `__atomic_compare_exchange_n()` 转换 PENDING → READY

**关键代码段**：
```c
// 关键：锁定任务的 fanout 以与协调器同步
// 防止的竞争条件：
//   1. 协调器正在向此任务的 fanout 列表添加消费者
//   2. 我们正在遍历 fanout 列表，可能会错过新消费者
while (PTO2_EXCHANGE(&task->fanout_lock, 1) != 0) {
    PTO2_SPIN_PAUSE();  // 让步以减少竞争
}

int32_t fanout_head = task->fanout_head;  // 获取锁后安全
int32_t current = fanout_head;

while (current > 0) {
    PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
    int32_t consumer_id = entry->task_id;
    int32_t consumer_slot = pto2_task_slot(sched, consumer_id);
    PTO2TaskDescriptor* consumer = pto2_sm_get_task(sched->sm_handle, consumer_id);

    // 原子地递增消费者的 fanin_refcount
    int32_t new_refcount = __atomic_add_fetch(&sched->fanin_refcount[consumer_slot], 1, __ATOMIC_SEQ_CST);

    int32_t fanin_count = __atomic_load_n(&consumer->fanin_count, __ATOMIC_ACQUIRE);

    if (new_refcount >= fanin_count) {
        // 尝试转换到 READY
        PTO2TaskState expected = PTO2_TASK_PENDING;
        if (__atomic_compare_exchange_n(&sched->task_state[consumer_slot], &expected, PTO2_TASK_READY,
                                         false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            pto2_scheduler_enqueue_ready_threadsafe(sched, consumer_id,
                                                     (PTO2WorkerType)consumer->worker_type, thread_ctx);
        }
    }

    current = entry->next_offset;
}

PTO2_STORE_RELEASE(&task->fanout_lock, 0);
```

### 1.3 性能特征

**时间复杂度**：
- **最佳情况**：O(1) - 无 fanout（叶任务）
- **最坏情况**：O(fanout) - 具有多个消费者的任务
- **平均情况**：O(average_fanout) - 通常较小（< 10）

**空间复杂度**：O(1) - 无额外分配

**同步开销**：
- **Fanout 锁**：带指数退避的自旋锁（PAUSE + yield）
- **原子操作**：每个消费者一次原子加
- **CAS 操作**：每次就绪转换一次 CAS

**潜在优化**：
1. **批处理**：一次处理多个完成事件（摊销锁开销）
2. **无锁 Fanout**：对 fanout 列表使用 RCU（Read-Copy-Update）
3. **依赖位掩码**：对小 fanin_count 使用位掩码代替计数器

---

## 2. 任务分发算法

### 2.1 就绪队列操作

**位置**: `pto_scheduler.c:69-91` - 基本队列操作

**入队操作**：
```c
bool pto2_ready_queue_push(PTO2ReadyQueue* queue, int32_t task_id) {
    if (queue->count >= queue->capacity) {
        return false;  // 队列满
    }

    queue->task_ids[queue->tail] = task_id;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;

    return true;
}
```

**出队操作**：
```c
int32_t pto2_ready_queue_pop(PTO2ReadyQueue* queue) {
    if (queue->count == 0) {
        return -1;  // 队列空
    }

    int32_t task_id = queue->task_ids[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;

    return task_id;
}
```

**设计特征**：
- **环形缓冲区**：避免内存分配，固定容量（65536默认）
- **线程安全包装**：外部 mutex/condvar 保护
- **非阻塞**：如果满/空立即返回

### 2.2 带选择性唤醒的线程安全推送

**位置**: `pto_scheduler.c:120-146` - `pto2_ready_queue_push_wake_min_clock()`

**算法**：
```
算法: PushAndWakeMinClock(task_id, worker_type)
输入: task_id, worker_type
输出: 任务已入队，具有最小时钟的工作器被唤醒

1. Acquire(queue_mutex[worker_type])

2. if queue.full():
3.     PrintError("就绪队列满！任务已丢弃")
4.     Release(queue_mutex[worker_type])
5.     return false

6. Push(task_id, queue)

7. // 查找具有最小模拟时钟的工作器
8. min_clock_worker ← -1
9. min_clock ← INFINITY
10. for worker_id in worker_range[worker_type]:
11.     if worker_waiting[worker_id] && worker_current_cycle[worker_id] < min_clock:
12.         min_clock ← worker_current_cycle[worker_id]
13.         min_clock_worker ← worker_id

14. // 仅唤醒具有最小时钟的工作器（公平调度）
15. if min_clock_worker != -1:
16.     pthread_cond_signal(&worker_cond[min_clock_worker])
17. else:
18.     // 没有工作器等待，广播给所有
19.     for worker_id in worker_range[worker_type]:
20.         if worker_waiting[worker_id]:
21.             pthread_cond_signal(&worker_cond[worker_id])

22. Release(queue_mutex[worker_type])
23. return true
```

**实现**：
```c
bool pto2_ready_queue_push_wake_min_clock(PTO2ReadyQueue* queue, int32_t task_id,
                                           pthread_mutex_t* mutex,
                                           volatile int64_t* worker_clocks,
                                           volatile bool* worker_waiting,
                                           pthread_cond_t* worker_conds,
                                           int32_t worker_start, int32_t worker_end) {
    pthread_mutex_lock(mutex);

    bool success = pto2_ready_queue_push(queue, task_id);

    if (success) {
        // 广播给所有等待的工作器
        // 每个工作器将检查自己是否具有最小时钟后再获取任务
        // 这确保具有最小时钟的工作器获得任务
        for (int32_t i = worker_start; i < worker_end; i++) {
            if (__atomic_load_n(&worker_waiting[i], __ATOMIC_ACQUIRE)) {
                pthread_cond_signal(&worker_conds[i]);
            }
        }
    } else {
        fprintf(stderr, "[错误] 就绪队列满！任务 %d 已丢弃！\n", task_id);
    }

    pthread_mutex_unlock(mutex);
    return success;
}
```

**设计原理**：
- **公平调度**：具有较小时钟（工作较少）的工作器获得优先级
- **广播策略**：向所有等待的工作器发信号，让它们自己检查时钟
- **低延迟**：避免在关键路径上计算 min_clock

### 2.3 带让步检查的线程安全弹出

**位置**: `pto_scheduler.c:172-206` - `pto2_ready_queue_pop_with_yield_check()`

**算法**：
```
算法: PopWithYieldCheck(worker_id, worker_clock)
输入: worker_id, worker_clock
输出: task_id 或 -1（关闭）

1. Acquire(queue_mutex[worker_type])

2. while true:
3.     // 队列空时等待
4.     while queue.empty() && !shutdown:
5.         pthread_cond_wait(cond, mutex)

6.     if shutdown && queue.empty():
7.         Release(queue_mutex[worker_type])
8.         return -1

9.     // 唤醒后，检查是否应该让步
10.    if should_yield_callback(worker_id, worker_clock):
11.        // 向其他工作器发信号并再次等待
12.        pthread_cond_broadcast(cond)
13.        pthread_cond_wait(cond, mutex)
14.        continue  // 重新检查条件

15.    break  // 我们可以获取任务

16. task_id ← Pop(queue)

17. Release(queue_mutex[worker_type])
18. return task_id
```

**让步条件**（在工作器中实现）：
```c
bool should_yield(void* ctx) {
    WorkerContext* worker = (WorkerContext*)ctx;
    int64_t my_clock = worker->current_cycle;

    // 检查是否有其他工作器具有更小的时钟
    for (int i = 0; i < num_workers; i++) {
        if (i != worker->worker_id && worker_waiting[i]) {
            int64_t other_clock = worker_current_cycle[i];
            if (other_clock < my_clock) {
                return true;  // 让步给工作较少的工作器
            }
        }
    }
    return false;  // 我们具有最小时钟，获取任务
}
```

**性能特征**：
- **无饥饿**：所有工作器最终都会获得任务（时钟保证公平性）
- **低竞争**：只有具有最小时钟的工作器继续，其他让步
- **自适应**：根据执行时间自动平衡负载

---

## 3. 流控算法

### 3.1 环形缓冲区管理

**位置**: `pto_scheduler.c:496-533` - `pto2_scheduler_advance_ring_pointers()`

**数据结构**：
```c
int32_t last_task_alive;      // 最旧的未 CONSUMED 任务
int32_t heap_tail;            // 可回收内存偏移
int32_t task_window_size;     // 窗口大小（2的幂，如16384）
int32_t task_window_mask;     // task_window_size - 1（用于快速取模）
```

**算法**：
```
算法: AdvanceRingPointers()
输入: 无（读取调度器状态）
输出: 更新 last_task_alive, heap_tail，同步到共享内存

1. current_task_index ← LoadAcquire(sm_header->current_task_index)

2. // 当任务为 CONSUMED 时推进 last_task_alive
3. while last_task_alive < current_task_index:
4.     slot ← last_task_alive & task_window_mask
5.     if task_state[slot] != CONSUMED:
6.         break  // 发现未消费任务，停止推进
7.     last_task_alive++

8. // 根据最后消费的任务的缓冲区更新 heap_tail
9. if last_task_alive > 0:
10.    last_consumed_id ← last_task_alive - 1
11.    last_consumed ← GetTaskDescriptor(last_consumed_id)
12.    if last_consumed->packed_buffer_end != NULL:
13.        heap_tail ← (int32_t)last_consumed->packed_buffer_end

14. // 写入共享内存以进行协调器流控
15. StoreRelease(sm_header->last_task_alive, last_task_alive)
16. StoreRelease(sm_header->heap_tail, heap_tail)
```

**实现**：
```c
void pto2_scheduler_advance_ring_pointers(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    int32_t current_task_index = PTO2_LOAD_ACQUIRE(&header->current_task_index);

    // 当该位置的任务为 CONSUMED 时推进 last_task_alive
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

    // 写入共享内存以进行协调器流控
    pto2_scheduler_sync_to_sm(sched);
}
```

### 3.2 协调器流控

**位置**：共享内存头（`pto_runtime2_types.h`）

**流控检查**（在协调器中）：
```c
// 提交任务前，检查窗口是否有空间
int32_t current_task_index = LoadAcquire(header->current_task_index);
int32_t last_task_alive = LoadAcquire(header->last_task_alive);

if (current_task_index - last_task_alive >= task_window_size) {
    // 窗口满，等待调度器推进 last_task_alive
    pthread_cond_wait(&header->window_cond, &header->window_mutex);
}
```

**设计特征**：
- **有界内存**：任务窗口大小限制并发任务（和内存使用）
- **背压**：窗口满时协调器阻塞
- **增量回收**：任务 CONSUMED 时 `heap_tail` 推进

### 3.3 内存回收

**位置**: `pto_scheduler.c:382-418` - `check_and_handle_consumed()`

**算法**：
```
算法: CheckAndHandleConsumed(task_id)
输入: task_id
输出: 如果所有引用都释放，转换 COMPLETED → CONSUMED

1. slot ← task_id & task_window_mask
2. task ← GetTaskDescriptor(task_id)

3. fanout_count ← LoadAcquire(task->fanout_count)
4. fanout_refcount ← LoadSeqCst(fanout_refcount[slot])

5. if fanout_refcount != fanout_count:
6.     return  // 尚未释放所有引用

7. // 使用 CAS 原子地转换 COMPLETED → CONSUMED
8. expected ← COMPLETED
9. if !CompareExchange(task_state[slot], &expected, CONSUMED):
10.    return  // CAS 失败 - 未 COMSUMED 或已 CONSUMED

11. tasks_consumed++

12. // 重置引用计数以供槽重用
13. StoreSeqCst(fanout_refcount[slot], 0)
14. StoreSeqCst(fanin_refcount[slot], 0)

15. // 尝试推进环形指针
16. if task_id == last_task_alive:
17.    AdvanceRingPointers()
```

**实现**：
```c
static void check_and_handle_consumed(PTO2SchedulerState* sched,
                                       int32_t task_id,
                                       PTO2TaskDescriptor* task) {
    int32_t slot = pto2_task_slot(sched, task_id);

    // 读取 fanout_count（由协调器设置，只增长）
    int32_t fanout_count = __atomic_load_n(&task->fanout_count, __ATOMIC_ACQUIRE);

    // 原子读取 fanout_refcount（由协调器和调度器线程修改）
    int32_t refcount = __atomic_load_n(&sched->fanout_refcount[slot], __ATOMIC_SEQ_CST);

    if (refcount != fanout_count) {
        return;  // 尚未释放所有引用
    }

    // 使用 CAS 原子地转换 COMPLETED → CONSUMED
    PTO2TaskState expected = PTO2_TASK_COMPLETED;
    if (!__atomic_compare_exchange_n(&sched->task_state[slot], &expected, PTO2_TASK_CONSUMED,
                                      false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        // CAS 失败 - 未 COMPLETED 或其他线程已转换
        return;
    }

    // 成功转换到 CONSUMED
    __atomic_fetch_add(&sched->tasks_consumed, 1, __ATOMIC_SEQ_CST);

    // 重置引用计数以供槽重用（环形缓冲区将重用此槽）
    __atomic_store_n(&sched->fanout_refcount[slot], 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&sched->fanin_refcount[slot], 0, __ATOMIC_SEQ_CST);

    // 尝试推进环形指针
    if (task_id == sched->last_task_alive) {
        pto2_scheduler_advance_ring_pointers(sched);
    }
}
```

**关键观察**：
1. **双重引用计数**：`fanout_refcount` 跟踪消费者完成和作用域结束
2. **CAS 转换**：确保只有一个线程执行 CONSUMED 转换
3. **立即回收**：引用计数重置为0以供槽重用
4. **指针推进**：仅当 `task_id == last_task_alive` 时尝试推进（顺序推进）

---

## 4. 调度器主循环

### 4.1 线程函数

**位置**: `pto_scheduler.c:838-934` - `pto2_scheduler_thread_func()`

**算法**：
```
算法: SchedulerThreadMain()
输入: SchedulerContext*
输出: NULL

1. // 首先等待所有工作器准备就绪
2. Acquire(startup_mutex)
3. while workers_ready < num_workers:
4.     pthread_cond_wait(startup_cond, startup_mutex)
5. scheduler_ready ← true
6. pthread_cond_broadcast(startup_cond)
7. Release(startup_mutex)

8. last_processed_task ← 0
9. while !shutdown:
10.    did_work ← false

11.    // === 步骤1：处理来自协调器的新任务 ===
12.    new_tasks ← ProcessNewTasksThreadsafe(&last_processed_task)
13.    if new_tasks > 0: did_work ← true

14.    // === 步骤2：处理来自工作器的完成事件 ===
15.    completions ← ProcessCompletions()
16.    if completions > 0: did_work ← true

17.    // === 步骤3：推进环形指针并同步到共享内存 ===
18.    AdvanceRingPointers()
19.    SyncToSharedMemory()  // 对协调器流控至关重要

20.    // === 步骤4：定期进度报告 ===
21.    if elapsed >= PROGRESS_REPORT_INTERVAL:
22.        PrintProgress(completed, consumed, submitted, last_alive)
23.        last_progress_time ← now

24.    // === 步骤5：检查是否全部完成 ===
25.    if IsDone():
26.        all_done ← true
27.        pthread_cond_broadcast(all_done_cond)
28.        break

29.    // 如果没有完成工作，短暂暂停以避免忙等待
30.    if !did_work:
31.        pthread_cond_timedwait(completion_cond, done_mutex, 1ms)

32. return NULL
```

**实现**：
```c
void* pto2_scheduler_thread_func(void* arg) {
    PTO2SchedulerContext* ctx = (PTO2SchedulerContext*)arg;
    PTO2SchedulerState* sched = ctx->scheduler;
    PTO2ThreadContext* thread_ctx = ctx->thread_ctx;

    // 首先等待所有工作器准备就绪
    pthread_mutex_lock(&thread_ctx->startup_mutex);
    while (thread_ctx->workers_ready < thread_ctx->num_workers) {
        pthread_cond_wait(&thread_ctx->startup_cond, &thread_ctx->startup_mutex);
    }
    thread_ctx->scheduler_ready = true;
    pthread_cond_broadcast(&thread_ctx->startup_cond);
    pthread_mutex_unlock(&thread_ctx->startup_mutex);

    int32_t last_processed_task = 0;

    while (!thread_ctx->shutdown) {
        bool did_work = false;

        // === 步骤1：处理来自协调器的新任务 ===
        int32_t new_tasks = process_new_tasks_threadsafe(sched, thread_ctx,
                                                          &last_processed_task);
        if (new_tasks > 0) {
            did_work = true;
        }

        // === 步骤2：处理来自工作器的完成事件 ===
        int32_t completions = pto2_scheduler_process_completions(ctx);
        if (completions > 0) {
            did_work = true;
        }

        // === 步骤3：推进环形指针并同步到共享内存 ===
        pto2_scheduler_advance_ring_pointers(sched);
        pto2_scheduler_sync_to_sm(sched);

        // === 步骤4：定期进度报告 ===
        // [进度报告代码省略]

        // === 步骤5：检查是否全部完成 ===
        if (pto2_scheduler_is_done(sched)) {
            pthread_mutex_lock(&thread_ctx->done_mutex);
            thread_ctx->all_done = true;
            pthread_cond_broadcast(&thread_ctx->all_done_cond);
            pthread_mutex_unlock(&thread_ctx->done_mutex);
            break;
        }

        // 如果没有完成工作，短暂暂停以避免忙等待
        if (!did_work) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 1000000;  // 1ms 超时
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }

            pthread_mutex_lock(&thread_ctx->done_mutex);
            pthread_cond_timedwait(&thread_ctx->completion_cond,
                                   &thread_ctx->done_mutex, &ts);
            pthread_mutex_unlock(&thread_ctx->done_mutex);
        }
    }

    return NULL;
}
```

**关键观察**：
1. **工作器启动同步**：调度器在启动前等待所有工作器
2. **轮询新任务**：定期检查 `current_task_index` 以获取新任务
3. **完成处理**：处理来自工作器的完成队列
4. **流控**：每次迭代将环形指针同步到共享内存
5. **关闭检测**：当协调器完成且所有任务消费时退出

---

## 5. 性能分析

### 5.1 调度开销分解

| 组件 | 操作 | 开销（每任务） | 频率 |
|-----------|-----------|---------------------|-----------|
| 依赖解析 | Fanout 列表遍历 | O(fanout) | 每次任务完成一次 |
| 原子操作 | Fanin 引用计数递增 | ~10-50 周期 | 每个消费者一次 |
| CAS 操作 | 状态转换 | ~50-100 周期 | 每次就绪转换一次 |
| 就绪队列 | 入队/出队 | ~20-30 周期 | 每次分发一次 |
| 环形指针 | 推进 | O(window_size) 最坏情况 | 每次任务完成 |
| Fanout 锁 | 自旋锁获取/释放 | ~100-1000 周期（竞争） | 每次任务完成一次 |

**总开销估计**：每任务约 ~100-2000 周期（取决于 fanout、竞争）

### 5.2 可扩展性特征

| 指标 | 值 | 备注 |
|--------|-------|-------|
| 最大工作器 | 128（64 CUBE + 64 VECTOR） | 可配置 |
| 就绪队列大小 | 每种工作器类型 65536 | 固定容量 |
| 任务窗口大小 | 16384（默认） | 2的幂，运行时可配置 |
| 最大 Fanout | 受 DepListPool 限制 | 65536 个条目 |
| 调度器线程 | 3（当 thread_num=4） | 固定 |

**可扩展性限制**：
- **内存**：`task_window_size * sizeof(TaskDescriptor)` ≈ 16384 × 340 字节 = ~5.5 MB
- **Fanout 列表**：`DEP_LIST_POOL_SIZE * sizeof(DepListEntry)` = 65536 × 8 字节 = ~512 KB
- **就绪队列**：`PTO2_READY_QUEUE_SIZE * 4 * sizeof(int32_t)` = 65536 × 4 × 4 = ~1 MB

**瓶颈**：
1. **Fanout 锁竞争**：协调器和调度器争夺 `fanout_lock`
2. **环形指针推进**：O(window_size) 最坏情况（除一个外所有任务都 CONSUMED）
3. **就绪队列互斥锁**：多个工作器时队列互斥锁竞争

### 5.3 优化机会

#### 短期（易于实现）
1. **批量依赖解析**：一次处理多个完成事件
   - **收益**：摊销锁开销，减少缓存未命中
   - **工作量**：低（修改 `on_task_complete_threadsafe`）

2. **每个工作器的就绪队列**：每个工作器一个队列而不是每种工作器类型
   - **收益**：消除队列互斥锁竞争
   - **工作量**：中等（改变队列结构，work-stealing 逻辑）

3. **自适应窗口大小**：根据内存压力动态调整 `task_window_size`
   - **收益**：更好的内存利用率
   - **工作量**：低（添加监控，运行时调整）

#### 中期（需要重构）
1. **无锁就绪队列**：使用 MPMC（多生产者多消费者）队列
   - **收益**：消除互斥锁开销
   - **工作量**：高（实现无锁数据结构）

2. **工作窃取**：工作器可以从其他工作器的队列窃取任务
   - **收益**：更好的负载均衡
   - **工作量**：高（设计窃取协议，避免饥饿）

3. **预测性调度**：根据估计执行时间排序就绪任务
   - **收益**：更好的关键路径利用率
   - **工作量**：中等（添加成本模型，排序就绪队列）

#### 长期（需要研究）
1. **分布式调度器**：多 NPU 协同调度及跨节点通信
   - **收益**：扩展到多个 NPU 设备
   - **工作量**：非常高（设计分布式协议）

2. **强化学习**：通过 RL 学习最优调度策略
   - **收益**：适应工作负载模式
   - **工作量**：非常高（集成 RL 框架，定义状态/动作空间）

3. **编译时调度**：PTO-AS 编译器中的静态调度优化
   - **收益**：零运行时开销
   - **工作量**：非常高（修改编译器，添加静态分析）

---

## 6. 总结

### 6.1 关键发现
1. **高效依赖解析**：O(fanout) 配合原子操作
2. **公平调度**：最小时钟唤醒确保负载均衡
3. **流控**：环形缓冲区窗口防止内存耗尽
4. **线程安全设计**：仔细使用锁、原子和 CAS

### 6.2 性能特征
- **低开销**：每任务约 ~100-2000 周期（取决于 fanout）
- **可扩展**：支持多达 128 个工作器
- **内存受限**：~7 MB 运行时内存（默认配置）

### 6.3 优化潜力
- **短期**：批处理、自适应窗口（~10-20% 改进）
- **中期**：无锁队列、工作窃取（~30-50% 改进）
- **长期**：分布式调度、RL（~2-5x 改进潜力）

---

**下一步**：参见 `03_执行器分析.md` 了解详细的执行器实现分析。
