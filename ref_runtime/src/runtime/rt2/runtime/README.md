# Rt2 runtime (single folder)

This directory contains all runtime sources for **rt2** in ref_runtime:

- **C++ runtime:** `runtime.cpp`, `runtime.h`, `pto2_dispatch_payload.h`
- **PTO2 core:** `pto_runtime2.c/h`, `pto_runtime2_types.h`
- **Shared memory / ring buffer:** `pto_shared_memory.c/h`, `pto_ring_buffer.c/h`
- **Scheduler / worker / orchestrator:** `pto_scheduler.c/h`, `pto_worker.c/h`, `pto_orchestrator.c/h`
- **Tensors / logical tensor / interval tree:** `pto_tensormap.c/h`, `pto_logical_tensor.c/h`, `pto_interval_tree.c/h`
- **Threaded interface (stub for device orchestration):** `pto_runtime2_threaded.h`, `pto_runtime2_threaded_stub.c`

Ref_runtime does not depend on the repository root `src/runtime2`. Build uses `build_config.py` with `include_dirs` / `source_dirs` pointing only at this `runtime` folder.
