/**
 * PTO Runtime System - Record & Replay Feature (ARCHIVED)
 * 
 * This file contains the archived Record & Replay functionality that was
 * removed from the main runtime due to conflicts with the sliding window
 * task management scheme.
 * 
 * The Record & Replay feature was designed to:
 * 1. Record the first iteration of a loop as a "fragment"
 * 2. Replay subsequent iterations by instantiating compact tasks from the template
 * 3. Reduce task creation overhead for repetitive loop patterns
 * 
 * REASON FOR ARCHIVAL:
 * The sliding window scheme (PTO_TASK_WINDOW_SIZE = 32K) wraps task IDs and
 * reuses task slots, which conflicts with the replay mechanism that assumes
 * task IDs are monotonically increasing and persistent.
 * 
 * DATE: January 2026
 */

#ifndef PTO_RECORD_REPLAY_ARCHIVED_H
#define PTO_RECORD_REPLAY_ARCHIVED_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Forward declarations
struct PTORuntime;

// =============================================================================
// Record & Replay Data Structures (ARCHIVED)
// =============================================================================

/**
 * Maximum arguments and fanout per task
 */
#define PTO_MAX_FANOUT_ARCHIVED  512
#define PTO_MAX_ARGS_ARCHIVED    16

/**
 * Task argument - either input or output tensor
 */
typedef struct {
    void*    raw_tensor;     // Base pointer to tensor data
    int64_t  row_offset;     // Row offset within tensor
    int64_t  col_offset;     // Column offset within tensor
    int64_t  rows;           // Number of rows in this region
    int64_t  cols;           // Number of columns in this region
    bool     is_output;      // True if this is an output argument
} TaskArgArchived;

/**
 * Cycle cost function pointer type
 */
typedef int64_t (*CycleCostFuncArchived)(void** args, int32_t num_args);

/**
 * Recorded task entry - immutable template for replay
 * Contains all data needed to replay a task without re-analyzing dependencies
 */
typedef struct RecordedTaskArchived {
    const char*  func_name;                              // InCore function name
    void*        func_ptr;                               // Function pointer
    CycleCostFuncArchived cycle_func;                    // Cycle cost function (for simulation)
    int32_t      buffer_size_bytes;                      // Buffer size estimation
    int32_t      buffer_size_with_reuse;                 // Buffer size with reuse
    int32_t      fanin;                                  // Initial fanin count (immutable)
    int32_t      internal_fanin;                         // Fanin from within same fragment (for replay)
    int32_t      fanout[PTO_MAX_FANOUT_ARCHIVED];        // Relative fanout offsets
    int32_t      fanout_count;                           // Number of fanouts
    TaskArgArchived args[PTO_MAX_ARGS_ARCHIVED];         // Arguments (with template regions)
    int32_t      num_args;                               // Number of arguments
    bool         is_cube;                                // True if requires cube unit (matmul)
} RecordedTaskArchived;

/**
 * Recorded output - for TensorMap replay
 */
typedef struct {
    void*        raw_tensor;
    int64_t      row_offset;
    int64_t      col_offset;
    int64_t      rows;
    int64_t      cols;
    int32_t      relative_producer;   // Offset from fragment base
} RecordedOutputArchived;

/**
 * Recorded fragment - a replayable task graph fragment
 */
typedef struct {
    RecordedTaskArchived* tasks;      // Array of recorded tasks
    int32_t         task_count;       // Number of tasks
    RecordedOutputArchived* outputs;  // Array of output registrations
    int32_t         output_count;     // Number of outputs
    const char*     fragment_name;    // Human-readable name
    int32_t         checksum;         // Simple checksum for validation
} RecordedFragmentArchived;

/**
 * Offset mode for loop replay
 */
typedef enum {
    OFFSET_NONE_ARCHIVED = 0,    // No offset adjustment
    OFFSET_ROW_ARCHIVED,         // Adjust row_offset only
    OFFSET_COL_ARCHIVED,         // Adjust col_offset only
    OFFSET_ROW_COL_ARCHIVED      // Adjust both row and col offset
} OffsetModeArchived;

/**
 * Loop replay context - manages record/replay for a single loop
 */
typedef struct {
    RecordedFragmentArchived* fragment;  // Recorded fragment (NULL before first record)
    int32_t           record_start;      // Start task_id for recording (-1 if not recording)
    int32_t           base_offset;       // Base offset from first recorded iteration
    int32_t           stride;            // Offset stride per iteration
    OffsetModeArchived offset_mode;      // How to adjust offsets during replay
    const char*       loop_name;         // For debugging
} LoopReplayCtxArchived;

/**
 * Compact task entry for replay (32 bytes - cache friendly)
 * 
 * PendingTask is ~2.8KB which causes severe cache thrashing during replay
 * (16K tasks = 735K cache lines touched). CompactTask is 32 bytes, fitting
 * ~2 entries per cache line (16K tasks = 8K cache lines = 90x better).
 */
typedef struct {
    RecordedTaskArchived* template_ref;  // 8 bytes: immutable template
    int32_t  resolved_fanin;             // 4 bytes: incremented when deps complete
    int32_t  offset_delta;               // 4 bytes: row offset for this replay
    int64_t  earliest_start_cycle;       // 8 bytes: for dependency-aware scheduling
    bool     is_complete;                // 1 byte
    bool     is_active;                  // 1 byte  
    int16_t  _padding;                   // 2 bytes alignment
} CompactTaskArchived;

// =============================================================================
// Record & Replay API (ARCHIVED)
// =============================================================================

/**
 * Global flag to enable/disable loop replay optimization
 */
// extern int pto_record_replay_enabled;

/**
 * Enable or disable loop replay optimization
 */
// void pto_set_record_replay(int enabled);

/**
 * Initialize loop replay context
 */
// void pto_loop_init(LoopReplayCtxArchived* ctx, const char* name, int32_t stride, OffsetModeArchived mode);

/**
 * Check if we should record this iteration (returns true) or replay (returns false)
 */
// bool pto_loop_should_record(struct PTORuntime* rt, LoopReplayCtxArchived* ctx, int32_t loop_idx);

/**
 * Finish recording the current iteration
 */
// void pto_loop_finish_record(struct PTORuntime* rt, LoopReplayCtxArchived* ctx);

/**
 * Cleanup loop replay context
 */
// void pto_loop_cleanup(LoopReplayCtxArchived* ctx);

/**
 * Validate that a task matches the recorded template (for debugging)
 */
// bool pto_loop_validate_task(LoopReplayCtxArchived* ctx, int32_t task_idx_in_fragment,
//                             int32_t loop_idx, TaskArgArchived* args, int32_t num_args);

/**
 * Record a range of tasks as a fragment
 */
// RecordedFragmentArchived* pto_fragment_record(struct PTORuntime* rt, int32_t start_id, int32_t end_id,
//                                               const char* name);

/**
 * Free a recorded fragment
 */
// void pto_fragment_free(RecordedFragmentArchived* fragment);

/**
 * Get fragment size in bytes
 */
// size_t pto_fragment_size(RecordedFragmentArchived* fragment);

/**
 * Replay the recorded fragment for this iteration
 * Platform-specific: uses appropriate ready queue
 */
// void pto_loop_replay(struct PTORuntime* rt, LoopReplayCtxArchived* ctx, int32_t loop_idx);

#endif // PTO_RECORD_REPLAY_ARCHIVED_H
