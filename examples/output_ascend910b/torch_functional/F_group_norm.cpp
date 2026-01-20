// PTO Program: F_group_norm
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class F_group_normKernel {
public:
    __aicore__ inline F_group_normKernel() {}
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

        // Loop fusion: 2 loop overheads saved

        // FUSED (1 ops): TLOAD
        // TLOAD: Operation

        // TROWSUM: reduction operation
        ReduceSum(mean, x, 8);

        // FUSED (1 ops): TDIVS
        Divs(mean, mean, 8.0f, 64);

        // TROWEXPANDSUB: Not implemented

        // FUSED (1 ops): TMUL
        Mul(sq_centered, centered, centered, 64);

        // TROWSUM: reduction operation
        ReduceSum(var, sq_centered, 8);

        // FUSED (3 ops): TDIVS; TADDS; TSQRT
        Divs(var, var, 8.0f, 64);
        Adds(var, var, 1e-05f, 64);
        Sqrt(std, var, 64);

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

extern "C" __global__ __aicore__ void F_group_norm_kernel(GM_ADDR input, GM_ADDR output) {
    F_group_normKernel op;
    op.Init(input, output);
    op.Process();
}