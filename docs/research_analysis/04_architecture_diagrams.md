# PTO-ISA Architecture Diagrams

This document contains visual architecture diagrams for the PTO-ISA system.

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Task State Machine](#2-task-state-machine)
3. [Scheduler Architecture](#3-scheduler-architecture)
4. [Memory Layout](#4-memory-layout)
5. [Execution Flow](#5-execution-flow)
6. [Dependency Graph](#6-dependency-graph)

---

## 1. System Architecture

```mermaid
graph TB
    subgraph "User Layer"
        A[Python DSL<br/>pto/pto_as] -->|PTO-AS text| B[PTO Compiler<br/>ptoas]
    end

    subgraph "Compiler Layer"
        B -->|C++ code| C[Binary Compiler<br/>ccec/gcc]
        C -->|Kernel binary| D[AICore Runtime]
        C -->|Runtime binary| E[AICPU Runtime]
    end

    subgraph "Runtime Layer"
        F[Python Runtime<br/>pto_runtime.py] -->|Task Graph| G[Host Runtime<br/>Graph/DeviceRunner]
        G -->|Shared Memory| H[Scheduler<br/>pto_scheduler.c]
        H -->|Handshake| I[AICPU Executor<br/>aicpu_executor.cpp]
    end

    subgraph "Hardware Layer"
        I -->|Dispatch| J[AICore Workers<br/>aicore_executor.cpp]
        J --> K[CUBE Units<br/>Matrix Ops]
        J --> L[VECTOR Units<br/>Element-wise Ops]
    end

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#ffe1f5
    style D fill:#e1ffe1
    style E fill:#e1ffe1
    style F fill:#f5e1ff
    style G fill:#f5e1ff
    style H fill:#ffe1e1
    style I fill:#ffe1e1
    style J fill:#e1e1ff
    style K fill:#e1e1ff
    style L fill:#e1e1ff
```

---

## 2. Task State Machine

```mermaid
stateDiagram-v2
    [*] --> PENDING: Task Submitted
    PENDING --> READY: fanin_refcount == fanin_count
    READY --> RUNNING: Dispatched to Worker
    RUNNING --> COMPLETED: Kernel Returns
    COMPLETED --> CONSUMED: fanout_refcount == fanout_count
    CONSUMED --> [*]: Slot Reclaimed

    note right of PENDING
        Waiting for dependencies
        fanin_refcount < fanin_count
    end note

    note right of READY
        All dependencies satisfied
        In ready queue
    end note

    note right of RUNNING
        Executing on AICore
        task_status = 1
    end note

    note right of COMPLETED
        Execution finished
        Output may be in use
    end note

    note right of CONSUMED
        Output fully consumed
        Buffers can be released
    end note
```

---

## 3. Scheduler Architecture

```mermaid
graph TB
    subgraph "Orchestrator Thread 3"
        A[Build Task Graph] --> B[Initialize Task Descriptors]
        B --> C[Set Initial State<br/>PENDING or READY]
        C --> D[Update Dependency Lists]
        D --> E[Signal Scheduler]
    end

    subgraph "Scheduler Threads 0/1/2"
        F[Process New Tasks] --> G[Initialize Ready Queues]
        G --> H[Wait for Completions]
        H --> I[Process Completions]
        I --> J{Task Completed?}
        J -->|Yes| K[Update fanin_refcount]
        K --> L{All Deps Met?}
        L -->|Yes| M[Transition to READY]
        M --> N[Enqueue to Ready Queue]
        N --> O[Wake Workers]
        L -->|No| P[Check Next Consumer]
        J -->|No| Q{More Tasks?}
        Q -->|Yes| H
        Q -->|No| R{All Done?}
        R -->|No| H
        R -->|Yes| S[Exit]
    end

    subgraph "AICore Workers"
        T[Wait for Task] --> U[Receive DispatchPayload]
        U --> V[Execute Kernel]
        V --> W[Write Completion]
        W --> T
    end

    E --> F
    O --> T
    W --> H

    style A fill:#ffe1e1
    style F fill:#e1f5ff
    style T fill:#e1ffe1
```

---

## 4. Memory Layout

```mermaid
graph TB
    subgraph "Shared Memory (GM)"
        A[Shared Memory Header]
        B[Task Descriptor Ring Buffer<br/>16,384 slots × 340 bytes]
        C[Dependency List Pool<br/>65,536 entries × 8 bytes]
        D[TensorMap<br/>Hash table for alias tracking]
        E[GM Heap<br/>Packed output buffers]
    end

    subgraph "Task Descriptor (340 bytes)"
        F[task_id: int32]
        G[kernel_id: int32]
        H[worker_type: int32]
        I[fanin_head: int32]
        J[fanin_count: int32]
        K[fanout_lock: volatile int32]
        L[fanout_head: volatile int32]
        M[fanout_count: volatile int32]
        N[packed_buffer_base: void*]
        O[packed_buffer_end: void*]
        P[output_offsets[16]: int32]
        Q[num_outputs: int32]
        R[func_ptr: void*]
        S[func_name: char*]
    end

    subgraph "GM Heap Layout"
        T[Task 0 Output<br/>offset 0-4096]
        U[Task 1 Output<br/>offset 4096-8192]
        V[Task 2 Output<br/>offset 8192-12288]
        W[Free Space<br/>offset 12288+]
    end

    A --> B
    A --> C
    A --> D
    A --> E

    B --> F
    E --> T

    style A fill:#ffe1e1
    style B fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#f5e1ff
    style E fill:#e1ffe1
```

---

## 5. Execution Flow

```mermaid
sequenceDiagram
    participant User as User Code
    participant Python as Python Runtime
    participant Host as Host Runtime
    participant Device as AICPU Device
    participant AICore as AICore Workers

    User->>Python: Graph.add_task()
    Python->>Python: Build task graph
    User->>Python: DeviceRunner.run(graph)

    Python->>Host: Compile kernels
    Host->>Host: ccec/aarch64-linux-gnu-gcc
    Host-->>Python: Binaries (.so/.bin)

    Python->>Host: Copy graph to device
    Host->>Device: rtKernelLaunch(aicpu)
    Device->>Device: Initialize executor

    par Scheduler Threads (0/1/2)
        Device->>Device: Process new tasks
        Device->>Device: Build ready queues
        loop Until all done
            Device->>AICore: Dispatch task via Handshake
            AICore->>AICore: Execute kernel
            AICore-->>Device: Completion signal
            Device->>Device: Update fanin_refcount
            Device->>Device: Enqueue newly ready tasks
        end
    and Orchestrator Thread (3)
        Device->>Device: Build task graph (device mode)
        Note over Device: Write task descriptors<br/>to shared memory
        Device->>Device: orchestrator_done = true
    end

    Device-->>Host: Completion
    Host-->>Python: Results
    Python-->>User: Output tensors

    Note over User,AICore: Total time: < 1s for 1K×1K GEMM
```

---

## 6. Dependency Graph Example

```mermaid
graph TB
    subgraph "Task Graph Example"
        A[Task 0: C = A @ B<br/>CUBE, fanin=0]
        B[Task 1: D = C + 1.0<br/>VECTOR, fanin=1]
        C[Task 2: E = D + 2.0<br/>VECTOR, fanin=1]
        D[Task 3: F = E @ G<br/>CUBE, fanin=1]

        A -->|fanout| B
        B -->|fanout| C
        C -->|fanout| D
    end

    subgraph "Execution Order"
        E1[1. Task 0 starts<br/>Immediately READY]
        E2[2. Task 0 completes<br/>→ Task 1 becomes READY]
        E3[3. Task 1 completes<br/>→ Task 2 becomes READY]
        E4[4. Task 2 completes<br/>→ Task 3 becomes READY]
        E5[5. Task 3 completes<br/>→ All DONE]
    end

    A -.->|triggers| E1
    B -.->|after Task 0| E2
    C -.->|after Task 1| E3
    D -.->|after Task 2| E4
    E5 -.->|final| E5

    style A fill:#e1ffe1
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#e1ffe1
    style E1 fill:#fff4e1
    style E2 fill:#fff4e1
    style E3 fill:#fff4e1
    style E4 fill:#fff4e1
    style E5 fill:#fff4e1
```

---

## 7. Ready Queue Distribution

```mermaid
graph LR
    subgraph "Scheduler"
        A[Task Completed]
        B[Dependency Resolution]
        C[Ready Queue Selection]
    end

    subgraph "Ready Queues"
        D[CUBE Queue<br/>65536 slots]
        E[VECTOR Queue<br/>65536 slots]
        F[AI_CPU Queue<br/>65536 slots]
        G[ACCELERATOR Queue<br/>65536 slots]
    end

    subgraph "Workers"
        H[CUBE Workers 0-31<br/>Waiting on CUBE Queue]
        I[VECTOR Workers 32-95<br/>Waiting on VECTOR Queue]
        J[AI_CPU Workers 96-127<br/>Waiting on AI_CPU Queue]
    end

    A --> B
    B --> C
    C -->|CUBE task| D
    C -->|VECTOR task| E
    C -->|AI_CPU task| F
    C -->|ACCEL task| G

    D --> H
    E --> I
    F --> J

    style A fill:#ffe1e1
    style B fill:#ffe1e1
    style C fill:#ffe1e1
    style D fill:#e1ffe1
    style E fill:#e1f5ff
    style F fill:#fff4e1
    style G fill:#f5e1ff
    style H fill:#e1ffe1
    style I fill:#e1f5ff
    style J fill:#fff4e1
```

---

## 8. Flow Control Mechanism

```mermaid
graph TB
    subgraph "Shared Memory Pointers"
        A[current_task_index<br/>Total tasks submitted]
        B[last_task_alive<br/>Oldest non-CONSUMED task]
        C[heap_tail<br/>Reclaimable memory offset]
    end

    subgraph "Orchestrator Check"
        D{Can submit task?}
        E[current_task_index -<br/>last_task_alive<br/>< window_size?]
        F[Submit task]
        G[BLOCK - Wait for<br/>scheduler to advance]
    end

    subgraph "Scheduler Advancement"
        H[Task becomes CONSUMED]
        I{Is sequential?}
        J[Advance last_task_alive++]
        K[Update heap_tail]
        L[Sync to shared memory]
    end

    subgraph "Flow Control Signal"
        M[Orchestrator wakes up]
        N[Resume submission]
    end

    A --> E
    B --> E
    E -->|Yes| F
    E -->|No| G
    F --> A

    H --> I
    I -->|task_id ==<br/>last_task_alive| J
    I -->|No| O[Skip]
    J --> K
    K --> L
    L --> M
    M --> N
    N --> D

    style A fill:#ffe1e1
    style B fill:#ffe1e1
    style C fill:#ffe1e1
    style D fill:#fff4e1
    style E fill:#fff4e1
    style F fill:#e1ffe1
    style G fill:#ffe1e1
    style H fill:#e1f5ff
    style I fill:#e1f5ff
    style J fill:#e1f5ff
    style K fill:#e1f5ff
    style L fill:#e1f5ff
    style M fill:#e1ffe1
    style N fill:#e1ffe1
```

---

## Diagram Usage

These diagrams can be viewed in any Mermaid-compatible viewer:
- **GitHub**: Automatically renders in markdown files
- **VS Code**: Install "Markdown Preview Mermaid Support" extension
- **Online**: Use [Mermaid Live Editor](https://mermaid.live)
- **Documentation**: Export to PNG/SVG for presentations

---

**Last Updated**: 2025-02-09
**Total Diagrams**: 8
**Formats**: Mermaid (text-based, version control friendly)
