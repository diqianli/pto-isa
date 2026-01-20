// PTO Program: attention_alibi
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class attention_alibiKernel {
public:
    __aicore__ inline attention_alibiKernel() {}
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

        // Loop fusion: 4 loop overheads saved

        // FUSED (4 ops): TLOAD; TLOAD; TLOAD; TLOAD
        // TLOAD: Operation
        // TLOAD: Operation
        // TLOAD: Operation
        // TLOAD: Operation

        // TMATMUL: scores = Q @ K
        Matmul(scores, Q, K, 8, 8);

        // FUSED (2 ops): TMULS; TADD
        Muls(scaled, scores, 0.35355339059327373f, 64);
        Add(biased_scores, scaled, alibi_bias, 64);

        // TROWSUM: reduction operation
        ReduceSum(row_sum, biased_scores, 8);

        // FUSED (1 ops): TDIVS
        Divs(row_sum, row_sum, 8.0f, 64);

        // TROWEXPANDSUB: Not implemented

        // FUSED (1 ops): TEXP
        Exp(exp_scores, shifted, 64);

        // TROWSUM: reduction operation
        ReduceSum(row_sum, exp_scores, 8);

        // TROWEXPANDDIV: Not implemented

        // TMATMUL: output = attn @ V
        Matmul(output, attn, V, 8, 8);

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

extern "C" __global__ __aicore__ void attention_alibi_kernel(GM_ADDR input, GM_ADDR output) {
    attention_alibiKernel op;
    op.Init(input, output);
    op.Process();
}