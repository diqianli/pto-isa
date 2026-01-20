// PTO Program: rmsnorm_tile
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rmsnorm_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 82,304 bytes (80.4 KB)
//   Total capacity (w/ reuse): 49,408 bytes (48.2 KB)
//   Reuse savings:            32,896 bytes (40.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   gamma                32x128     f32     16384   [  1,  10]           -
//   result               32x128     f32     16384   [ 10,  11]           <- x
//   row_mean             32x1       f32       128   [  5,   8]           -
//   row_rsqrt            32x1       f32       128   [  8,   9]           <- row_sum
//   row_sum              32x1       f32       128   [  3,   5]           -
//   x                    32x128     f32     16384   [  0,   9]           -
//   x_norm               32x128     f32     16384   [  9,  10]           <- x_sq
//   x_sq                 32x128     f32     16384   [  2,   3]           -
//
// BUFFER REUSE MAP:
//   row_rsqrt reuses buffer of row_sum
//   x_norm reuses buffer of x_sq
//   result reuses buffer of x
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class rmsnorm_tileKernel {
public:
    __aicore__ inline rmsnorm_tileKernel() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
        inputGm.SetGlobalBuffer((__gm__ float*)input);
        outputGm.SetGlobalBuffer((__gm__ float*)output);
        pipe.InitBuffer(inQueueX, 1, 8 * 8 * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, 8 * 8 * sizeof(float));
    }

    __aicore__ inline void Process() {
        CopyIn(); Compute(); CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, inputGm, 64);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Loop fusion: 5 loop overheads saved

        // FUSED (3 ops): TLOAD; TLOAD; TMUL
        // TLOAD: Operation
        // TLOAD: Operation
        Mul(x_sq, x, x, 64);

        // TROWSUM: reduction operation
        ReduceSum(row_sum, x_sq, 32);

        int inv_cols = 0.0078125;

        // FUSED (1 ops): TMULS
        Muls(row_mean, row_sum, inv_colsf, 64);

        int eps = 1e-05;

        // FUSED (2 ops): TADDS; TRSQRT
        Adds(row_mean, row_mean, epsf, 64);
        Rsqrt(row_rsqrt, row_mean, 64);

        // FUSED (3 ops): TROWEXPANDMUL; TMUL; TSTORE
        BroadcastMul(x_norm, x, row_rsqrt, 64, 8);  // row-wise broadcast multiply
        Mul(result, x_norm, gamma, 64);
        // TSTORE: Operation

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(outputGm, yLocal, 64);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
};

extern "C" __global__ __aicore__ void rmsnorm_tile_kernel(GM_ADDR input, GM_ADDR output) {
    rmsnorm_tileKernel op;
    op.Init(input, output);
    op.Process();
}