// PTO Program: score_to_weight
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class score_to_weightKernel {
public:
    __aicore__ inline score_to_weightKernel() {}
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

        // Loop fusion: 0 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TROWSUM: reduction operation
        ReduceSum(row_sum, scores, 8);

        // FUSED (1 ops): TDIVS
        Divs(row_sum, row_sum, 8.0f, 64);

        // TROWEXPANDSUB: Not implemented

        // FUSED (1 ops): TEXP
        Exp(exp_scores, shifted, 64);

        // TROWSUM: reduction operation
        ReduceSum(row_sum, exp_scores, 8);

        // TROWEXPANDDIV: Not implemented

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

extern "C" __global__ __aicore__ void score_to_weight_kernel(GM_ADDR input, GM_ADDR output) {
    score_to_weightKernel op;
    op.Init(input, output);
    op.Process();
}