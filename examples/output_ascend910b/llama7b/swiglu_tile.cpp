// PTO Program: swiglu_tile
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: swiglu_tile
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     8
//   Total capacity (no reuse): 131,072 bytes (128.0 KB)
//   Total capacity (w/ reuse): 65,536 bytes (64.0 KB)
//   Reuse savings:            65,536 bytes (50.0%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   exp_neg_gate         32x128     f32     16384   [  3,   4]           -
//   gate                 32x128     f32     16384   [  0,   6]           -
//   gate_silu            32x128     f32     16384   [  6,   7]           <- one_plus_exp
//   neg_gate             32x128     f32     16384   [  2,   3]           -
//   one_plus_exp         32x128     f32     16384   [  4,   5]           <- neg_gate
//   result               32x128     f32     16384   [  7,   8]           <- gate
//   sigmoid_gate         32x128     f32     16384   [  5,   6]           <- exp_neg_gate
//   up                   32x128     f32     16384   [  1,   7]           -
//
// BUFFER REUSE MAP:
//   one_plus_exp reuses buffer of neg_gate
//   sigmoid_gate reuses buffer of exp_neg_gate
//   gate_silu reuses buffer of one_plus_exp
//   result reuses buffer of gate
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class swiglu_tileKernel {
public:
    __aicore__ inline swiglu_tileKernel() {}
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

        // Loop fusion: 8 loop overheads saved

        // FUSED (9 ops): TLOAD; TLOAD; TNEG; TEXP; TADDS; TRECIP; TMUL; TMUL; TSTORE
        // TLOAD: Operation
        // TLOAD: Operation
        Neg(neg_gate, gate, 64);
        Exp(exp_neg_gate, neg_gate, 64);
        Adds(one_plus_exp, exp_neg_gate, 1.0f, 64);
        Reciprocal(sigmoid_gate, one_plus_exp, 64);
        Mul(gate_silu, gate, sigmoid_gate, 64);
        Mul(result, gate_silu, up, 64);
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

extern "C" __global__ __aicore__ void swiglu_tile_kernel(GM_ADDR input, GM_ADDR output) {
    swiglu_tileKernel op;
    op.Init(input, output);
    op.Process();
}