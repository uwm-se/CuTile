#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

// Raw SIMT hand-written GEMM - shared memory tiling, NO Tensor Cores
// Each thread computes one element of C using scalar FMA instructions.
// This is the baseline showing performance without hardware matrix units.

__global__ void gemm_kernel_raw_simt_bfloat16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    const int TILE_SIZE = 32;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ __nv_bfloat16 tileA[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 tileB[TILE_SIZE][TILE_SIZE];
    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int aRow = row;
        int aCol = tile * TILE_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (aRow < M && aCol < K) ?
            A[aRow * K + aCol] : __float2bfloat16(0.0f);

        int bRow = tile * TILE_SIZE + threadIdx.y;
        int bCol = col;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && bCol < N) ?
            B[bRow * N + bCol] : __float2bfloat16(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += __bfloat162float(tileA[threadIdx.y][k]) * __bfloat162float(tileB[k][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(sum);
    }
}

torch::Tensor gemm_raw_simt(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    TORCH_CHECK(A.size(1) == B.size(0));
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "Raw SIMT GEMM expects BF16 input");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    gemm_kernel_raw_simt_bfloat16<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_raw_simt, "Raw SIMT GEMM (BF16, shared mem tiling, no Tensor Cores)");
}
