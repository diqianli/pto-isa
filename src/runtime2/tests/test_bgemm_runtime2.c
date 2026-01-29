/**
 * BGEMM Test for Runtime2
 * 
 * Tests the runtime2 system with a batched GEMM pattern:
 * For each batch:
 *   For each tile (m, n):
 *     For k in K:
 *       gemm_tile: P[m,n] = A[m,k] * B[k,n]
 *       tile_add:  C[m,n] += P[m,n]
 * 
 * Usage:
 *   ./test_bgemm_runtime2 [batch] [m_tiles] [n_tiles] [k_tiles] [mode]
 *   mode: 0 = single-threaded (default), 1 = multi-threaded
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../pto_runtime2.h"
#include "../pto_runtime2_sim.h"
#include "../pto_runtime2_threaded.h"

// Test configuration
#define TEST_BATCH    4
#define TEST_M_TILES  4
#define TEST_N_TILES  4
#define TEST_K_TILES  4

// Default worker counts
#define DEFAULT_CUBE_WORKERS   4
#define DEFAULT_VECTOR_WORKERS 4

// =============================================================================
// BGEMM Orchestration Parameters
// =============================================================================

typedef struct {
    int batch;
    int m_tiles;
    int n_tiles;
    int k_tiles;
    float* A;
    float* B;
    float* C;
    float* P;
    int task_count;
} BgemmParams;

// =============================================================================
// BGEMM Orchestration Function (for multi-threaded mode)
// =============================================================================

static void bgemm_orchestration(PTO2Runtime* rt, void* arg) {
    BgemmParams* p = (BgemmParams*)arg;
    
    p->task_count = 0;
    
    for (int b = 0; b < p->batch; b++) {
        pto2_rt_scope_begin(rt);  // Batch scope
        
        for (int m = 0; m < p->m_tiles; m++) {
            for (int n = 0; n < p->n_tiles; n++) {
                pto2_rt_scope_begin(rt);  // Tile scope
                
                for (int k = 0; k < p->k_tiles; k++) {
                    // Task indices for dependency tracking
                    int a_idx = b * (p->m_tiles * p->k_tiles) + m * p->k_tiles + k;
                    int b_idx = b * (p->k_tiles * p->n_tiles) + k * p->n_tiles + n;
                    int c_idx = b * (p->m_tiles * p->n_tiles) + m * p->n_tiles + n;
                    
                    // gemm_tile: P = A * B (Cube operation)
                    PTO2TaskParam gemm_params[3] = {
                        PTO2_INPUT(p->A, a_idx, 128),
                        PTO2_INPUT(p->B, b_idx, 128),
                        PTO2_OUTPUT(p->P, c_idx, 128)
                    };
                    pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL,
                                        "gemm_tile", gemm_params, 3);
                    p->task_count++;
                    
                    // tile_add: C += P (Vector operation)
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(p->C, c_idx, 128),
                        PTO2_INPUT(p->P, c_idx, 128),
                        PTO2_OUTPUT(p->C, c_idx, 128)
                    };
                    pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, NULL,
                                        "tile_add", add_params, 3);
                    p->task_count++;
                }
                
                pto2_rt_scope_end(rt);  // End tile scope
            }
        }
        
        pto2_rt_scope_end(rt);  // End batch scope
    }
}

// =============================================================================
// Single-Threaded Test (Original)
// =============================================================================

static int run_single_threaded_test(int batch, int m_tiles, int n_tiles, int k_tiles) {
    int total_tasks = batch * m_tiles * n_tiles * k_tiles * 2;
    
    printf("=== BGEMM Runtime2 Test (Single-Threaded) ===\n");
    printf("Configuration:\n");
    printf("  Batch:    %d\n", batch);
    printf("  M tiles:  %d\n", m_tiles);
    printf("  N tiles:  %d\n", n_tiles);
    printf("  K tiles:  %d\n", k_tiles);
    printf("  Total tasks: %d\n\n", total_tasks);
    
    // Create runtime in simulation mode
    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_SIMULATE);
    if (!rt) {
        fprintf(stderr, "Failed to create runtime\n");
        return 1;
    }
    
    // Allocate dummy tensors
    float* A = (float*)calloc(1024 * 1024, sizeof(float));
    float* B = (float*)calloc(1024 * 1024, sizeof(float));
    float* C = (float*)calloc(1024 * 1024, sizeof(float));
    float* P = (float*)calloc(1024 * 1024, sizeof(float));
    
    // Measure orchestration time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    printf("Phase 1: Building task graph...\n");
    
    int task_count = 0;
    
    for (int b = 0; b < batch; b++) {
        pto2_rt_scope_begin(rt);
        
        for (int m = 0; m < m_tiles; m++) {
            for (int n = 0; n < n_tiles; n++) {
                pto2_rt_scope_begin(rt);
                
                for (int k = 0; k < k_tiles; k++) {
                    int a_idx = b * (m_tiles * k_tiles) + m * k_tiles + k;
                    int b_idx = b * (k_tiles * n_tiles) + k * n_tiles + n;
                    int c_idx = b * (m_tiles * n_tiles) + m * n_tiles + n;
                    
                    PTO2TaskParam gemm_params[3] = {
                        PTO2_INPUT(A, a_idx, 128),
                        PTO2_INPUT(B, b_idx, 128),
                        PTO2_OUTPUT(P, c_idx, 128)
                    };
                    pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL,
                                        "gemm_tile", gemm_params, 3);
                    task_count++;
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128),
                        PTO2_INPUT(P, c_idx, 128),
                        PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, NULL,
                                        "tile_add", add_params, 3);
                    task_count++;
                }
                
                pto2_rt_scope_end(rt);
            }
        }
        
        pto2_rt_scope_end(rt);
    }
    
    pto2_rt_orchestration_done(rt);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double orch_time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                          (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    printf("  Submitted %d tasks\n", task_count);
    printf("  Orchestration time: %.3f ms (%.2f tasks/ms)\n\n", 
           orch_time_ms, task_count / orch_time_ms);
    
    // Run simulation
    printf("Phase 2: Running simulation...\n");
    
    PTO2SimConfig sim_config = {
        .num_vector_cores = DEFAULT_VECTOR_WORKERS,
        .num_cube_cores = DEFAULT_CUBE_WORKERS,
        .trace_enabled = true,
        .trace_filename = "bgemm_runtime2_trace.json"
    };
    
    PTO2SimState* sim = pto2_sim_create(&sim_config);
    if (!sim) {
        fprintf(stderr, "Failed to create simulation state\n");
        pto2_runtime_destroy(rt);
        return 1;
    }
    
    int64_t total_cycles = pto2_sim_run(sim, rt);
    
    printf("  Total cycles: %lld\n", (long long)total_cycles);
    printf("  Trace saved to: %s\n\n", sim_config.trace_filename);
    
    // Print summary
    printf("=== Summary ===\n");
    printf("  Tasks:        %d\n", task_count);
    printf("  Orch time:    %.3f ms\n", orch_time_ms);
    printf("  Throughput:   %.2f tasks/ms\n", task_count / orch_time_ms);
    printf("  Sim cycles:   %lld\n", (long long)total_cycles);
    
    pto2_runtime_print_stats(rt);
    
    // Cleanup
    pto2_sim_destroy(sim);
    pto2_runtime_destroy(rt);
    free(A);
    free(B);
    free(C);
    free(P);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}

// =============================================================================
// Multi-Threaded Test
// =============================================================================

static int run_multi_threaded_test(int batch, int m_tiles, int n_tiles, int k_tiles) {
    int total_tasks = batch * m_tiles * n_tiles * k_tiles * 2;
    
    printf("=== BGEMM Runtime2 Test (Multi-Threaded) ===\n");
    printf("Configuration:\n");
    printf("  Batch:    %d\n", batch);
    printf("  M tiles:  %d\n", m_tiles);
    printf("  N tiles:  %d\n", n_tiles);
    printf("  K tiles:  %d\n", k_tiles);
    printf("  Total tasks: %d\n", total_tasks);
    printf("  CUBE workers:   %d\n", DEFAULT_CUBE_WORKERS);
    printf("  VECTOR workers: %d\n\n", DEFAULT_VECTOR_WORKERS);
    
    // Create threaded runtime in simulation mode
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(
        DEFAULT_CUBE_WORKERS, DEFAULT_VECTOR_WORKERS, true);
    
    if (!rt) {
        fprintf(stderr, "Failed to create threaded runtime\n");
        return 1;
    }
    
    // Allocate dummy tensors
    float* A = (float*)calloc(1024 * 1024, sizeof(float));
    float* B = (float*)calloc(1024 * 1024, sizeof(float));
    float* C = (float*)calloc(1024 * 1024, sizeof(float));
    float* P = (float*)calloc(1024 * 1024, sizeof(float));
    
    // Setup parameters
    BgemmParams params = {
        .batch = batch,
        .m_tiles = m_tiles,
        .n_tiles = n_tiles,
        .k_tiles = k_tiles,
        .A = A,
        .B = B,
        .C = C,
        .P = P,
        .task_count = 0
    };
    
    // Measure total time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    printf("Running multi-threaded execution...\n");
    
    // Run with multi-threading (orchestrator in separate thread)
    pto2_runtime_run_threaded(rt, bgemm_orchestration, &params);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                           (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    int64_t total_cycles = pto2_runtime_get_total_cycles(rt);
    
    // Print summary
    printf("\n=== Summary ===\n");
    printf("  Tasks:        %d\n", params.task_count);
    printf("  Total time:   %.3f ms\n", total_time_ms);
    printf("  Throughput:   %.2f tasks/ms\n", params.task_count / total_time_ms);
    printf("  Sim cycles:   %lld\n", (long long)total_cycles);
    
    // Print threaded stats
    pto2_runtime_print_threaded_stats(rt);
    
    // Write trace
    pto2_runtime_write_trace(rt, "bgemm_runtime2_threaded_trace.json");
    
    // Cleanup
    pto2_runtime_destroy_threaded(rt);
    free(A);
    free(B);
    free(C);
    free(P);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int batch = TEST_BATCH;
    int m_tiles = TEST_M_TILES;
    int n_tiles = TEST_N_TILES;
    int k_tiles = TEST_K_TILES;
    int mode = 0;  // 0 = single-threaded, 1 = multi-threaded
    
    // Parse optional args
    if (argc > 1) batch = atoi(argv[1]);
    if (argc > 2) m_tiles = atoi(argv[2]);
    if (argc > 3) n_tiles = atoi(argv[3]);
    if (argc > 4) k_tiles = atoi(argv[4]);
    if (argc > 5) mode = atoi(argv[5]);
    
    if (mode == 0) {
        return run_single_threaded_test(batch, m_tiles, n_tiles, k_tiles);
    } else {
        return run_multi_threaded_test(batch, m_tiles, n_tiles, k_tiles);
    }
}
