#include <immintrin.h>
#include <omp.h>
#include <pthread.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, fmt, ##args);
#define DEBUG_MSG(str) std::cout << str << "\n";
#else
#define DEBUG_PRINT(fmt, args...)
#define DEBUG_MSG(str)
#endif  // DEBUG

#ifdef TIMING
#include <ctime>
#define TIMING_START(arg)          \
    struct timespec __start_##arg; \
    clock_gettime(CLOCK_MONOTONIC, &__start_##arg);
#define TIMING_END(arg)                                                                       \
    {                                                                                         \
        struct timespec __temp_##arg, __end_##arg;                                            \
        double __duration_##arg;                                                              \
        clock_gettime(CLOCK_MONOTONIC, &__end_##arg);                                         \
        if ((__end_##arg.tv_nsec - __start_##arg.tv_nsec) < 0) {                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec - 1;              \
            __temp_##arg.tv_nsec = 1000000000 + __end_##arg.tv_nsec - __start_##arg.tv_nsec;  \
        } else {                                                                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec;                  \
            __temp_##arg.tv_nsec = __end_##arg.tv_nsec - __start_##arg.tv_nsec;               \
        }                                                                                     \
        __duration_##arg = __temp_##arg.tv_sec + (double)__temp_##arg.tv_nsec / 1000000000.0; \
        printf("%s took %lfs.\n", #arg, __duration_##arg);                                    \
    }
#else
#define TIMING_START(arg)
#define TIMING_END(arg)
#endif  // TIMING

#define block_size 64
#define half_bs 32
#define div_block 2
const int INF = ((1 << 30) - 1);

struct edge_t {
    int src;
    int dst;
    int w;
};

int blk_idx(int r, int c, int nblocks);

void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus);

__global__ void proc_1_glob(int *blk_dist, int k, int nblocks);
__global__ void proc_2_glob(int *blk_dist, int s, int k, int nblocks);
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int nblocks);

int main(int argc, char **argv) {
    assert(argc == 3);

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    FILE *input_file;
    FILE *output_file;
    int ncpus = omp_get_max_threads();
    int V, E;
    edge_t *edge;
    int *dist;
    int VP;
    int nblocks;
    int *blk_dist;
    int *blk_dist_dev;

    TIMING_START(hw3_1);

    /* input */
    TIMING_START(input);
    input_file = fopen(input_filename, "rb");
    assert(input_file);
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);
    edge = (edge_t *)malloc(sizeof(edge_t) * E);
    fread(edge, sizeof(edge_t), E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);
    fclose(input_file);
    DEBUG_PRINT("vertices: %d\nedges: %d\n", V, E);
    TIMING_END(input);

    /* calculate */
    TIMING_START(calculate);
    nblocks = (int)ceilf(float(V) / block_size);
    VP = nblocks * block_size;
    blk_dist = (int *)malloc(sizeof(int) * VP * VP);

    for (int i = 0; i < VP; i++) {
        for (int j = 0; j < VP; j++) {
            if (i == j)
                blk_dist[blk_idx(i, j, nblocks)] = 0;
            else
                blk_dist[blk_idx(i, j, nblocks)] = INF;
        }
    }

    for (int i = 0; i < E; i++) {
        blk_dist[blk_idx(edge[i].src, edge[i].dst, nblocks)] = edge[i].w;
    }

    cudaHostRegister(blk_dist, sizeof(int) * VP * VP, cudaHostRegisterDefault);
    cudaMalloc(&blk_dist_dev, sizeof(int) * VP * VP);
    cudaMemcpy(blk_dist_dev, blk_dist, sizeof(int) * VP * VP, cudaMemcpyHostToDevice);

    dim3 blk(block_size / div_block, block_size / div_block);
    for (int k = 0, nk = nblocks - 1; k < nblocks; k++, nk--) {
        /* Phase 1 */
        proc_1_glob<<<1, blk>>>(blk_dist_dev, k, nblocks);
        /* Phase 2 */
        proc_2_glob<<<nblocks, blk>>>(blk_dist_dev, 0, k, nblocks);
        /* Phase 3 */
        proc_3_glob<<<dim3(nblocks, nblocks), blk>>>(blk_dist_dev, 0, 0, k, nblocks);
    }

    cudaMemcpy(blk_dist, blk_dist_dev, sizeof(int) * VP * VP, cudaMemcpyDeviceToHost);

    /* Copy result to dist */
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            dist[i * V + j] = min(blk_dist[blk_idx(i, j, nblocks)], INF);
        }
    }

    TIMING_END(calculate);

    /* output */
    TIMING_START(output);
    output_file = fopen(output_filename, "w");
    assert(output_file);
    fwrite(dist, sizeof(int), V * V, output_file);
    fclose(output_file);
    TIMING_END(output);
    TIMING_END(hw3_1);

    /* finalize */
    free(edge);
    free(dist);
    free(blk_dist);
    cudaFree(blk_dist_dev);
    return 0;
}

int blk_idx(int r, int c, int nblocks) {
    return ((r / block_size) * nblocks + (c / block_size)) * block_size * block_size + (r % block_size) * block_size + (c % block_size);
}

void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus) {
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = s_i; i < e_i; i++) {
        for (int j = s_j; j < e_j; j++) {
            int *ik_ptr = blk_dist + (i * nblocks + k) * block_size * block_size;
            int *ij_ptr = blk_dist + (i * nblocks + j) * block_size * block_size;
            int *kj_ptr = blk_dist + (k * nblocks + j) * block_size * block_size;
            for (int b = 0; b < block_size; b++) {
                for (int r = 0; r < block_size; r++) {
#pragma omp simd
                    for (int c = 0; c < block_size; c++) {
                        ij_ptr[r * block_size + c] = std::min(ij_ptr[r * block_size + c], ik_ptr[r * block_size + b] + kj_ptr[b * block_size + c]);
                    }
                }
            }
        }
    }
}

__global__ void proc_1_glob(int *blk_dist, int k, int nblocks) {
    __shared__ int k_k_sm[block_size][block_size];

    int r = threadIdx.y;
    int c = threadIdx.x;
    int *k_k_ptr = blk_dist + (k * nblocks + k) * (block_size * block_size);
    int tmp;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * half_bs][c + cc * half_bs] = k_k_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs];
        }
    }
    __syncthreads();

    // #pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                tmp = k_k_sm[r + rr * half_bs][b] + k_k_sm[b][c + cc * half_bs];
                if (tmp < k_k_sm[r + rr * half_bs][c + cc * half_bs])
                    k_k_sm[r + rr * half_bs][c + cc * half_bs] = tmp;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs] = k_k_sm[r + rr * half_bs][c + cc * half_bs];
        }
    }
}
__global__ void proc_2_glob(int *blk_dist, int s, int k, int nblocks) {
    __shared__ int i_k_sm[block_size][block_size];
    __shared__ int k_j_sm[block_size][block_size];
    __shared__ int k_k_sm[block_size][block_size];

    int i = s + blockIdx.x;
    int j = s + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int *i_k_ptr = blk_dist + (i * nblocks + k) * (block_size * block_size);
    int *k_j_ptr = blk_dist + (k * nblocks + j) * (block_size * block_size);
    int *k_k_ptr = blk_dist + (k * nblocks + k) * (block_size * block_size);
    int tmp_i_k, tmp_k_j;

    if (i == k)
        return;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_k_sm[r + rr * half_bs][c + cc * half_bs] = i_k_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs];
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_j_sm[r + rr * half_bs][c + cc * half_bs] = k_j_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs];
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * half_bs][c + cc * half_bs] = k_k_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs];
        }
    }
    __syncthreads();

    // #pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                tmp_i_k = i_k_sm[r + rr * half_bs][b] + k_k_sm[b][c + cc * half_bs];
                if (tmp_i_k < i_k_sm[r + rr * half_bs][c + cc * half_bs])
                    i_k_sm[r + rr * half_bs][c + cc * half_bs] = tmp_i_k;
                tmp_k_j = k_k_sm[r + rr * half_bs][b] + k_j_sm[b][c + cc * half_bs];
                if (tmp_k_j < k_j_sm[r + rr * half_bs][c + cc * half_bs])
                    k_j_sm[r + rr * half_bs][c + cc * half_bs] = tmp_k_j;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_k_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs] = i_k_sm[r + rr * half_bs][c + cc * half_bs];
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_j_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs] = k_j_sm[r + rr * half_bs][c + cc * half_bs];
        }
    }
}
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int nblocks) {
    __shared__ int i_k_sm[block_size][block_size];
    __shared__ int k_j_sm[block_size][block_size];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int *i_k_ptr = blk_dist + (i * nblocks + k) * (block_size * block_size);
    int *i_j_ptr = blk_dist + (i * nblocks + j) * (block_size * block_size);
    int *k_j_ptr = blk_dist + (k * nblocks + j) * (block_size * block_size);
    int loc[div_block][div_block], tmp;

    if (i == k || j == k)
        return;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_k_sm[r + rr * half_bs][c + cc * half_bs] = i_k_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs];
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_j_sm[r + rr * half_bs][c + cc * half_bs] = k_j_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs];
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            loc[rr][cc] = i_j_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs];
        }
    }

    // #pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                tmp = i_k_sm[r + rr * half_bs][b] + k_j_sm[b][c + cc * half_bs];
                if (tmp < loc[rr][cc])
                    loc[rr][cc] = tmp;
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_j_ptr[(r + rr * half_bs) * block_size + c + cc * half_bs] = loc[rr][cc];
        }
    }
}