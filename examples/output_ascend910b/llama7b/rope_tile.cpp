// PTO Program: rope_tile
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: rope_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     6
//   Total capacity (no reuse): 98,304 bytes (96.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            32,768 bytes (33.3%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   cos_pos              32x128     f32     16384   [  1,   3]           -
//   result               32x128     f32     16384   [  5,   6]           <- x
//   sin_pos              32x128     f32     16384   [  2,   4]           -
//   x                    32x128     f32     16384   [  0,   4]           -
//   x_cos                32x128     f32     16384   [  3,   5]           -
//   x_sin                32x128     f32     16384   [  4,   5]           <- cos_pos
//
// BUFFER REUSE MAP:
//   x_sin reuses buffer of cos_pos
//   result reuses buffer of x
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class rope_tileKernel {
public:
    __aicore__ inline rope_tileKernel() {}
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

        // Loop fusion: 6 loop overheads saved

        // FUSED (7 ops): TLOAD; TLOAD; TLOAD; TMUL; TMUL; TADD; TSTORE
        // TLOAD: Operation
        // TLOAD: Operation
        // TLOAD: Operation
        Mul(x_cos, x, cos_pos, 64);
        Mul(x_sin, x, sin_pos, 64);
        Add(result, x_cos, x_sin, 64);
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

extern "C" __global__ __aicore__ void rope_tile_kernel(GM_ADDR input, GM_ADDR output) {
    rope_tileKernel op;
    op.Init(input, output);
    op.Process();
}