// PTO Program: flash_attn_softmax_update
// Function Type: InCore (tile-level computation)
// ======================================================================
// TILE BUFFER ANALYSIS: flash_attn_softmax_update
// ======================================================================
//
// SUMMARY:
//   Total tiles declared:     12
//   Total capacity (no reuse): 51,456 bytes (50.2 KB)
//   Total capacity (w/ reuse): 34,048 bytes (33.2 KB)
//   Reuse savings:            17,408 bytes (33.8%)
//
// TILE DETAILS:
//   Name                 Shape      Type   Bytes    Liveness [write,read]   Reuse
//   --------------------------------------------------------------------------------
//   l_new                64x1       f32       256   [ 11,  13]           -
//   l_prev               64x1       f32       256   [  2,   9]           -
//   l_scaled             64x1       f32       256   [  9,  11]           <- m_diff
//   m_cur                64x1       f32       256   [  3,   4]           -
//   m_diff               64x1       f32       256   [  7,   8]           <- m_cur
//   m_new                64x1       f32       256   [  4,  12]           -
//   m_prev               64x1       f32       256   [  1,   7]           -
//   p_block              64x64      f32     16384   [  6,  14]           <- s_block
//   p_rowsum             64x1       f32       256   [ 10,  11]           <- l_prev
//   s_block              64x64      f32     16384   [  0,   5]           -
//   s_shifted            64x64      f32     16384   [  5,   6]           -
//   scale_old            64x1       f32       256   [  8,  15]           <- m_prev
//
// BUFFER REUSE MAP:
//   p_block reuses buffer of s_block
//   scale_old reuses buffer of m_prev
//   m_diff reuses buffer of m_cur
//   l_scaled reuses buffer of m_diff
//   p_rowsum reuses buffer of l_prev
//
// ======================================================================

// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class flash_attn_softmax_updateKernel {
public:
    __aicore__ inline flash_attn_softmax_updateKernel() {}
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

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // FUSED (2 ops): TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation

        // TROWMAX: reduction max operation
        ReduceMax(m_cur, s_block, 64);

        // FUSED (1 ops): TMAX
        Max(m_new, m_prev, m_cur, 64);

        // FUSED (2 ops): TROWEXPANDSUB; TEXP
        BroadcastSub(s_shifted, s_block, m_new, 64, 8);  // row-wise broadcast subtract
        Exp(p_block, s_shifted, 64);

        // FUSED (3 ops): TSUB; TEXP; TMUL
        Sub(m_diff, m_prev, m_new, 64);
        Exp(scale_old, m_diff, 64);
        Mul(l_scaled, scale_old, l_prev, 64);

        // TROWSUM: reduction operation
        ReduceSum(p_rowsum, p_block, 64);

        // FUSED (3 ops): TADD; TSTORE; TSTORE
        Add(l_new, l_scaled, p_rowsum, 64);
        // TSTORE: Operation
        // TSTORE: Operation

        // FUSED (1 ops): TSTORE
        // TSTORE: Operation

        // FUSED (1 ops): TSTORE
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

extern "C" __global__ __aicore__ void flash_attn_softmax_update_kernel(GM_ADDR input, GM_ADDR output) {
    flash_attn_softmax_updateKernel op;
    op.Init(input, output);
    op.Process();
}