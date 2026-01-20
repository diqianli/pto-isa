// PTO Program: dynamic_softmax
// Function Type: InCore (tile-level computation)
// Auto-generated Ascend C code from PTO ISA Compiler
// Target: Huawei Ascend 910B (Da Vinci Architecture)
#include "kernel_operator.h"

using namespace AscendC;

class dynamic_softmaxKernel {
public:
    __aicore__ inline dynamic_softmaxKernel() {}
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

        int tile_rows = 8;

        int zero = 0;

        for (int tile_idx = 0; tile_idx < num_full_tiles; tile_idx += 1) {

            // FUSED (1 ops): TLOAD
            // TLOAD: Operation

            // TROWMAX: reduction max operation
            ReduceMax(row_max, x, 8);

            // FUSED (2 ops): TROWEXPANDSUB; TEXP
            BroadcastSub(x_shifted, x, row_max, 64, 8);  // row-wise broadcast subtract
            Exp(exp_x, x_shifted, 64);

            // TROWSUM: reduction operation
            ReduceSum(row_sum, exp_x, 8);

            // FUSED (2 ops): TROWEXPANDDIV; TSTORE
            BroadcastDiv(result, exp_x, row_sum, 64, 8);  // row-wise broadcast divide
            // TSTORE: Operation

        }

        int has_tail = (tail_rows > zero) ? 1 : 0;

        if (has_tail) {

            // FUSED (1 ops): TLOAD
            // TLOAD: Operation

            // TROWMAX: reduction max operation
            ReduceMax(row_max, x, 8);

            // FUSED (2 ops): TROWEXPANDSUB; TEXP
            BroadcastSub(x_shifted, x, row_max, 64, 8);  // row-wise broadcast subtract
            Exp(exp_x, x_shifted, 64);

            // TROWSUM: reduction operation
            ReduceSum(row_sum, exp_x, 8);

            // FUSED (2 ops): TROWEXPANDDIV; TSTORE
            BroadcastDiv(result, exp_x, row_sum, 64, 8);  // row-wise broadcast divide
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

extern "C" __global__ __aicore__ void dynamic_softmax_kernel(GM_ADDR input, GM_ADDR output) {
    dynamic_softmaxKernel op;
    op.Init(input, output);
    op.Process();
}