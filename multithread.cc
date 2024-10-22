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

#define block_size 40
const int INF = ((1 << 30) - 1);

struct edge_t {
    int src;
    int dst;
    int w;
};

int blk_idx(int r, int c, int nblocks);

void proc(int *blk_dist, int s_i, int e_i, int s_j, int e_j, int k, int nblocks, int ncpus);

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

    TIMING_START(hw3_1);

    /* input */
    TIMING_START(input);
    input_file = fopen(input_filename, "rb");
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

#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = 0; i < VP; i++) {
        for (int j = 0; j < VP; j++) {
            if(i == j)
                blk_dist[blk_idx(i, j, nblocks)] = 0;
            else
                blk_dist[blk_idx(i, j, nblocks)] = INF;
        }
    }

#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared)
    for (int i = 0; i < E; i++) {
        blk_dist[blk_idx(edge[i].src, edge[i].dst, nblocks)] = edge[i].w;
    }

    for (int k = 0; k < nblocks; k++) {
        /* Phase 1 */
        proc(blk_dist, k, k + 1, k, k + 1, k, nblocks, ncpus);
        /* Phase 2 */
        proc(blk_dist, k, k + 1, 0, k, k, nblocks, ncpus);
        proc(blk_dist, k, k + 1, k + 1, nblocks, k, nblocks, ncpus);
        proc(blk_dist, 0, k, k, k + 1, k, nblocks, ncpus);
        proc(blk_dist, k + 1, nblocks, k, k + 1, k, nblocks, ncpus);
        /* Phase 3 */
        proc(blk_dist, 0, k, 0, k, k, nblocks, ncpus);
        proc(blk_dist, 0, k, k + 1, nblocks, k, nblocks, ncpus);
        proc(blk_dist, k + 1, nblocks, 0, k, k, nblocks, ncpus);
        proc(blk_dist, k + 1, nblocks, k + 1, nblocks, k, nblocks, ncpus);
    }

    /* Copy result to dist */
#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) collapse(2)
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            dist[i * V + j] = blk_dist[blk_idx(i, j, nblocks)] > INF ? INF : blk_dist[blk_idx(i, j, nblocks)];
        }
    }

    TIMING_END(calculate);

    /* output */
    TIMING_START(output);
    output_file = fopen(output_filename, "w");
    fwrite(dist, sizeof(int), V * V, output_file);
    fclose(output_file);
    TIMING_END(output);
    TIMING_END(hw3_1);

    /* finalize */
    free(edge);
    free(dist);
    free(blk_dist);
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