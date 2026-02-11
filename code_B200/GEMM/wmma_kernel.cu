#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

// ============================================================================
// Optimized WMMA GEMM - Tensor Cores + Shared Memory + cp.async pipeline
// BF16 input, FP32 accumulation, BF16 output
//
// Key optimizations:
//   1. Shared memory tiling with cooperative loading
//   2. cp.async for async global->shared copies (bypasses registers)
//   3. Software pipeline: 3-stage (triple buffer) to overlap load + compute
//   4. Register tiling: each warp computes 2x2 WMMA tiles (32x32 output)
//   5. Shared memory padding to eliminate bank conflicts
//   6. __launch_bounds__ to enforce occupancy >= 2 blocks/SM
// ============================================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Each warp computes 2x2 WMMA tiles = 32x32 output
#define WARP_TILES_M 2
#define WARP_TILES_N 2
#define WARP_M (WARP_TILES_M * WMMA_M)  // 32
#define WARP_N (WARP_TILES_N * WMMA_N)  // 32

// Thread block: 4x4 warps = 16 warps = 512 threads
#define WARPS_M 4
#define WARPS_N 4
#define NUM_WARPS (WARPS_M * WARPS_N)  // 16

// Block output tile
#define BLOCK_M (WARPS_M * WARP_M)  // 128
#define BLOCK_N (WARPS_N * WARP_N)  // 128
#define BLOCK_K 32  // 2 WMMA_K steps per shared memory tile

// Padding for: (1) bank conflict avoidance, (2) 16B alignment for cp.async
// cp.async.cg requires 16-byte aligned addresses
// A row = BLOCK_K=32 BF16 -> pad to 40 elements (80 bytes, 16B-aligned)
// B row = BLOCK_N=128 BF16 -> pad to 136 elements (272 bytes, 16B-aligned)
#define PAD_A 8
#define PAD_B 8
#define LDA_S (BLOCK_K + PAD_A)   // 40  -> 80 bytes/row, 16B-aligned
#define LDB_S (BLOCK_N + PAD_B)   // 136 -> 272 bytes/row, 16B-aligned

#define NUM_STAGES 2

// ---- cp.async helpers (sm_80+) ----
__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(global_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// launch_bounds: 512 threads, target >=2 blocks/SM -> compiler limits regs to ~64
__global__ __launch_bounds__(512, 2)
void gemm_wmma_optimized(
    const void* __restrict__ A_raw,
    const void* __restrict__ B_raw,
    void* __restrict__ C_raw,
    int M, int N, int K
) {
    const __nv_bfloat16* A = reinterpret_cast<const __nv_bfloat16*>(A_raw);
    const __nv_bfloat16* B = reinterpret_cast<const __nv_bfloat16*>(B_raw);
    __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(C_raw);

    // Triple-buffered shared memory with padding
    __shared__ __nv_bfloat16 smem_A[NUM_STAGES][BLOCK_M][LDA_S];
    __shared__ __nv_bfloat16 smem_B[NUM_STAGES][BLOCK_K][LDB_S];

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warp_m = warpId / WARPS_N;
    const int warp_n = warpId % WARPS_N;

    const int tid = threadIdx.x;
    const int num_k_tiles = K / BLOCK_K;

    // Accumulators: 2x2 WMMA tiles per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // ==== Helper: async load a tile of A and B using cp.async ====
    auto load_tile_async = [&](int stage, int k_offset) {
        // Load A: each thread loads 16 bytes (8 BF16 elements)
        {
            const int loads_per_row = BLOCK_K / 8;  // 4
            int row = tid / loads_per_row;
            int col = (tid % loads_per_row) * 8;
            int gr = block_row + row;
            int gc = k_offset + col;

            const __nv_bfloat16* src = A + gr * K + gc;
            __nv_bfloat16* dst = &smem_A[stage][row][col];

            if (gr < M && gc + 7 < K) {
                cp_async_16B(dst, src);
            } else {
                for (int e = 0; e < 8; e++) {
                    dst[e] = (gr < M && gc + e < K)
                        ? src[e] : __float2bfloat16(0.0f);
                }
            }
        }

        // Load B: 32x128 tile
        {
            const int loads_per_row = BLOCK_N / 8;  // 16
            int row = tid / loads_per_row;
            int col = (tid % loads_per_row) * 8;
            int gr = k_offset + row;
            int gc = block_col + col;

            const __nv_bfloat16* src = B + gr * N + gc;
            __nv_bfloat16* dst = &smem_B[stage][row][col];

            if (gr < K && gc + 7 < N) {
                cp_async_16B(dst, src);
            } else {
                for (int e = 0; e < 8; e++) {
                    dst[e] = (gr < K && gc + e < N)
                        ? src[e] : __float2bfloat16(0.0f);
                }
            }
        }
    };

    // ==== Fill pipeline: load first NUM_STAGES tiles ====
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        if (s < num_k_tiles) {
            load_tile_async(s, s * BLOCK_K);
        }
        cp_async_commit();
    }

    // ==== Main K-loop: compute stage[i], then load future tile into freed buffer ====
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int stage = kt % NUM_STAGES;

        // Wait for this stage's data to arrive
        cp_async_wait<NUM_STAGES - 1>();
        __syncthreads();

        // ==== WMMA compute from current stage (BEFORE issuing new loads) ====
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> b_frag;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int wi = 0; wi < WARP_TILES_M; wi++) {
                int a_row = warp_m * WARP_M + wi * WMMA_M;

                wmma::load_matrix_sync(a_frag,
                    &smem_A[stage][a_row][kk], LDA_S);

                #pragma unroll
                for (int wj = 0; wj < WARP_TILES_N; wj++) {
                    int b_col = warp_n * WARP_N + wj * WMMA_N;

                    wmma::load_matrix_sync(b_frag,
                        &smem_B[stage][kk][b_col], LDB_S);

                    wmma::mma_sync(c_frag[wi][wj], a_frag, b_frag, c_frag[wi][wj]);
                }
            }
        }

        __syncthreads();  // all warps done reading from stage before overwriting

        // Now safe to start loading future tile into this (now-freed) buffer
        if (kt + NUM_STAGES < num_k_tiles) {
            load_tile_async(stage, (kt + NUM_STAGES) * BLOCK_K);
        }
        cp_async_commit();
    }

    // ==== Store: FP32 -> BF16 via per-warp scratch ====
    float* warp_scratch = reinterpret_cast<float*>(&smem_A[0][0][0])
                        + warpId * (WMMA_M * WMMA_N);

    #pragma unroll
    for (int wi = 0; wi < WARP_TILES_M; wi++) {
        #pragma unroll
        for (int wj = 0; wj < WARP_TILES_N; wj++) {
            int out_row = block_row + warp_m * WARP_M + wi * WMMA_M;
            int out_col = block_col + warp_n * WARP_N + wj * WMMA_N;

            wmma::store_matrix_sync(warp_scratch, c_frag[wi][wj],
                                    WMMA_N, wmma::mem_row_major);

            #pragma unroll
            for (int e = laneId; e < WMMA_M * WMMA_N; e += 32) {
                int r = e / WMMA_N;
                int c_idx = e % WMMA_N;
                int gr = out_row + r;
                int gc = out_col + c_idx;
                if (gr < M && gc < N) {
                    C[gr * N + gc] = __float2bfloat16(warp_scratch[e]);
                }
            }
            __syncwarp();
        }
    }
}

torch::Tensor gemm_wmma(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    TORCH_CHECK(A.size(1) == B.size(0));
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "WMMA GEMM expects BF16 input");

    int M = A.size(0), K = A.size(1), N = B.size(1);

    TORCH_CHECK(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % BLOCK_K == 0,
                "Dims must be multiples of 128 (M,N) and 32 (K). M=", M, " N=", N, " K=", K);

    auto C = torch::empty({M, N}, A.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_WARPS * 32);  // 512 threads

    gemm_wmma_optimized<<<grid, block, 0, stream>>>(
        A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_wmma,
          "WMMA GEMM (BF16, Tensor Cores, cp.async pipeline, shared mem)");
}
