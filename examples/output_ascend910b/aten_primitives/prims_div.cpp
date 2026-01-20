// PTO Program: prims_div
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class prims_divKernel {
public:
    __aicore__ inline prims_divKernel() {}
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

        int tile_size = 4096;

        int zero = 0;

        for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

            // FUSED (4 ops): TLOAD; TLOAD; TDIV; TSTORE
            // TLOAD: Operation
            // TLOAD: Operation
            Div(result, x, y, 64);
            // TSTORE: Operation

        }

        int has_tail = (tail_elements > zero) ? 1 : 0;

        if (has_tail) {

            // FUSED (4 ops): TLOAD; TLOAD; TDIV; TSTORE
            // TLOAD: Operation
            // TLOAD: Operation
            Div(result, x, y, 64);
            // TSTORE: Operation

        }

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

extern "C" __global__ __aicore__ void prims_div_kernel(GM_ADDR input, GM_ADDR output) {
    prims_divKernel op;
    op.Init(input, output);
    op.Process();
}