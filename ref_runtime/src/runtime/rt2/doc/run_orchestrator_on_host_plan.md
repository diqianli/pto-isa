# Rt2: run_orchestrator_on_host 实现计划

## 1. 目标

- 在 rt2 中增加选项：**run_orchestrator_on_host**（编排在 Host CPU 上运行）与现有 **run_orchestrator_on_device_AICPU**（编排在 AICPU 线程 3 上运行）二选一。
- 保持 rt2 所有 API 以及 Scheduler–Worker 关系不变；仅改变 **Orchestrator 与 Scheduler 之间共享内存的分配方式**：
  1. 通过 **runtime 接口** 分配该 shared memory；
  2. 所有 Orchestrator 与 Scheduler 共同访问的数据结构都放在这块 shared memory 中。

## 2. 当前行为（device AICPU）

- Host 用 `device_malloc(PTO2_SM_SIZE)` 分配设备侧 SM，`set_pto2_gm_sm_ptr(dev_sm)` 传给 runtime。
- AICPU 共 4 线程：3 个 Scheduler（thread 0/1/2）+ 1 个 Orchestrator（thread 3）。
- Thread 3 调用 `aicpu_orchestration_entry(sm_ptr, args, arg_count)`，其中 `sm_ptr` 为**设备指针**；内部 `pto2_sm_create_from_buffer(sm_ptr, ...)`、`pto2_runtime_create_from_sm`，在同一块设备 SM 上提交任务。
- Scheduler 从同一块设备 SM 读取 TaskDescriptor 等，逻辑不变。

## 3. 目标行为（host CPU）

- **共享内存仍为一块**：布局与 device 模式一致（PTO2SharedMemoryHeader + TaskDescriptor 环 + DepListPool），Scheduler 仍只读该块。
- **分配**：由 **runtime 接口** 分配（例如在 init 或单独 API 中调用 `host_api.device_malloc(sm_size)`），得到设备侧 SM 指针并保存（如 `set_pto2_gm_sm_ptr`）。
- **Host 编排**：
  1. 在 Host 上分配一块与 PTO2 SM **同布局、同大小**的 **host 镜像**（host mirror）。
  2. 在 Host 上执行编排逻辑：用 `pto2_sm_create_from_buffer(host_mirror, ...)` + `pto2_runtime_create_from_sm` 在 **host mirror** 上提交任务（与 device 编排相同的 C 逻辑，仅 buffer 指针不同）。
  3. 编排结束后 `copy_to_device(dev_sm, host_mirror, sm_size)`，将 host mirror 整块拷贝到已分配好的设备 SM。
  4. 之后 Scheduler/Worker 仅访问设备 SM，行为与 device 编排完全一致。
- **线程**：AICPU 仅起 3 个 Scheduler 线程（不再起 Orchestrator 线程）；不再调用 `aicpu_orchestration_entry`。

## 4. 实现项清单

### 4.1 命令行与选项

- **run_example.py**：增加 `--orchestrator`，取值 `host_cpu` | `device_aicpu`；仅当 `-r rt2` 时有效；默认可为 `device_aicpu`（与当前 `-u` 行为一致）。
- **code_runner.py**：增加 `orchestrator_location`（或等价的 `run_orchestrator_on_host`），由 run_example 传入；用于分支：host 编排 vs device 编排。

### 4.2 Runtime 接口（共享内存由 runtime 分配）

- **分配**：在 rt2 路径下，当 `run_orchestrator_on_host` 时，在 **init** 阶段由 runtime 调用 `host_api.device_malloc(sm_size)` 分配设备 SM，并 `set_pto2_gm_sm_ptr(dev_sm)`；不再由 Python 单独 `device_malloc` 再 `set_pto2_gm_sm_ptr`。
- **大小**：暴露 `get_pto2_sm_size()`（或等价常量/API），供 Python 分配 host mirror 时使用。
- **Host 编排一步**：增加 `run_host_orchestration(host_mirror_ptr, args, arg_count)`（或由 init 内部在拿到 host_mirror 后调用）：  
  调用 `host_orchestration_entry(host_mirror_ptr, args, arg_count)`，然后 `host_api.copy_to_device(get_pto2_gm_sm_ptr(), host_mirror_ptr, get_pto2_sm_size())`。  
  这样“所有 orchestrator 与 scheduler 共同访问的数据”都先写在 host mirror，再整块搬到已由 runtime 分配的 device SM 中。

### 4.3 Host 编排入口与逻辑复用

- **host_orchestration_entry(void* host_sm_mirror, uint64_t* args, int arg_count)**：与 `aicpu_orchestration_entry` 相同的图构建逻辑（同一 DAG、同一 pto2_rt_submit_task 序列），仅第一个参数为 host 可写 buffer。
- 实现方式：在 `example_aicpu_orchestration_entry.cpp`（或同一模块）中抽成共用函数 `run_example_graph(void* sm_ptr, uint64_t* args, int arg_count)`，`aicpu_orchestration_entry` 与 `host_orchestration_entry` 均调用它；保证 layout 与 device 模式一致。
- `host_orchestration_entry` 需在 **host 侧**可调用（链接进 host_runtime 或由 host 加载的 SO 导出），以便 init/run_host_orchestration 调用。

### 4.4 Init / Launch 分支

- **run_orchestrator_on_host == true**：
  - init：若 rt2，则用 runtime 分配 device SM（见上），`set_pto2_gm_sm_ptr`；`set_orch_args(args, count)`；不启动 AICPU 上的编排（不依赖 thread 3 建图）。
  - Python 分配 host mirror（大小为 get_pto2_sm_size()），调用 `run_host_orchestration(host_mirror, args, count)`（内部完成 host 编排 + copy_to_device）。
  - launch：AICPU 线程数 = 3（仅 Scheduler），不再为 Orchestrator 留 thread 3。
- **run_orchestrator_on_device_AICPU**：保持现有逻辑；Python 仍可保留当前 `device_malloc` + `set_pto2_gm_sm_ptr` 的用法，或统一改为由 runtime 分配（见下“可选统一”）。

### 4.5 可选统一（推荐）

- 为满足“共享内存一律由 runtime 分配”：
  - device 编排时也可在 init 中由 runtime 分配 device SM（`host_api.device_malloc`），并 `set_pto2_gm_sm_ptr`，Python 不再单独 `device_malloc` SM；仅需在 launch 前把该指针交给 thread 3（已通过 runtime 保存，无需改 API）。
- 这样两种模式都满足：**先通过 runtime 接口分配 shared memory，再在各自位置（host 或 device）写入同一 layout，scheduler 只读该块**。

## 5. 文件与 API 变更摘要

| 位置 | 变更 |
|------|------|
| run_example.py | 增加 `--orchestrator host_cpu \| device_aicpu`；传入 code_runner。 |
| code_runner.py | 增加 `orchestrator_location` / `run_orchestrator_on_host`；host 时分配 host mirror、调 run_host_orchestration、AICPU 线程数=3；device 时保持 4 线程、现有 device 分配与 set_pto2_gm_sm_ptr。 |
| ref_runtime C API / Runtime 类 | `get_pto2_sm_size()`；`run_host_orchestration(host_mirror_ptr, args, count)`；init 内（rt2 + run_orchestrator_on_host）分配 device SM 并 set_pto2_gm_sm_ptr。 |
| rt2 host (runtime_maker.cpp) | init 时若 rt2 且 run_orchestrator_on_host，则分配 device SM；支持 run_host_orchestration 调用 host_orchestration_entry + copy_to_device。 |
| rt2 aicpu (example_aicpu_orchestration_entry.cpp) | 抽出共用函数；增加 `host_orchestration_entry`；该符号需在 host 可链接（例如同一实现编入 host 或导出给 host 调用）。 |

## 6. 数据流小结

- **Device 编排**：runtime 分配（或当前由 Python 分配）device SM → thread 3 在 device 上写该 SM → Scheduler 读该 SM。
- **Host 编排**：runtime 分配 device SM；Host 分配 host mirror；Host 上 host_orchestration_entry(host_mirror, args) 写 mirror；copy_to_device(dev_sm, host_mirror)；Scheduler 读 dev_sm（与 device 编排一致）。

Scheduler–Worker 关系与现有 API 保持不变；仅 Orchestrator 所在位置与 SM 的“写入路径”（host mirror → copy vs device 直接写）不同。
