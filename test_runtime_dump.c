/**
 * Test program for pto_runtime_dump function
 */

#include "pto_runtime.h"
#include "pto_runtime.c"  // Include implementation for standalone test

int main() {
    // Initialize runtime
    PTORuntime rt;
    pto_runtime_init(&rt);
    
    // Allocate dummy buffers
    float input[64];
    float temp_rowmax[8];
    float temp_shifted[64];
    float temp_exp[64];
    float temp_rowsum[8];
    float output[64];
    
    // Build task graph (simulating softmax)
    
    // Task 0: rowmax(input) -> temp_rowmax
    int32_t t0 = pto_task_alloc(&rt, "rowmax", NULL);
    pto_task_add_input(&rt, t0, input, 0, 0, 8, 8);
    pto_task_add_output(&rt, t0, temp_rowmax, 0, 0, 8, 1);
    pto_task_submit(&rt, t0);
    
    // Task 1: rowexpandsub(input, temp_rowmax) -> temp_shifted
    int32_t t1 = pto_task_alloc(&rt, "rowexpandsub", NULL);
    pto_task_add_input(&rt, t1, input, 0, 0, 8, 8);
    pto_task_add_input(&rt, t1, temp_rowmax, 0, 0, 8, 1);
    pto_task_add_output(&rt, t1, temp_shifted, 0, 0, 8, 8);
    pto_task_submit(&rt, t1);
    
    // Task 2: elem_exp(temp_shifted) -> temp_exp
    int32_t t2 = pto_task_alloc(&rt, "elem_exp", NULL);
    pto_task_add_input(&rt, t2, temp_shifted, 0, 0, 8, 8);
    pto_task_add_output(&rt, t2, temp_exp, 0, 0, 8, 8);
    pto_task_submit(&rt, t2);
    
    // Task 3: rowsum(temp_exp) -> temp_rowsum
    int32_t t3 = pto_task_alloc(&rt, "rowsum", NULL);
    pto_task_add_input(&rt, t3, temp_exp, 0, 0, 8, 8);
    pto_task_add_output(&rt, t3, temp_rowsum, 0, 0, 8, 1);
    pto_task_submit(&rt, t3);
    
    // Task 4: rowexpanddiv(temp_exp, temp_rowsum) -> output
    int32_t t4 = pto_task_alloc(&rt, "rowexpanddiv", NULL);
    pto_task_add_input(&rt, t4, temp_exp, 0, 0, 8, 8);
    pto_task_add_input(&rt, t4, temp_rowsum, 0, 0, 8, 1);
    pto_task_add_output(&rt, t4, output, 0, 0, 8, 8);
    pto_task_submit(&rt, t4);
    
    // Dump to stdout
    printf("\n========== RUNTIME DUMP (stdout) ==========\n");
    pto_runtime_dump_stdout(&rt);
    
    // Dump to file
    pto_runtime_dump(&rt, "runtime_state.txt");
    
    // Shutdown
    pto_runtime_shutdown(&rt);
    
    return 0;
}
