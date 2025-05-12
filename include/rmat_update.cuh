#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <iostream>

#define MAX_EDGES 10000000
#define MAX_WEIGHT 16
#define BLOCK_SIZE 256

typedef unsigned int vid_t;
typedef float real_t;

struct Edge {
    vid_t u, v;
};

__device__ __host__ inline Edge make_edge(vid_t u, vid_t v) {
    return {u, v};
}

struct gpu_graph {
    vid_t vertices;
    size_t edges;
    size_t* d_xadj = nullptr;
    vid_t* d_adj = nullptr;
    real_t* d_weight = nullptr;

    void allocate(size_t vertices_, size_t edges_) {
        vertices = vertices_;
        edges = edges_;
        cudaMalloc(&d_xadj, sizeof(size_t) * (vertices + 1));
        cudaMalloc(&d_adj, sizeof(vid_t) * edges);
        cudaMalloc(&d_weight, sizeof(real_t) * edges);
    }

    void free_all() {
        if (d_xadj) cudaFree(d_xadj);
        if (d_adj) cudaFree(d_adj);
        if (d_weight) cudaFree(d_weight);
    }
};

// ============ RMAT on GPU =============

__global__ void generate_rmat_edges_gpu(Edge* edges, real_t* weights,
                                        size_t n_edges, vid_t vertices,
                                        double a, double b, double c, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_edges) return;

    curandState rng;
    curand_init(seed + tid, 0, 0, &rng);

    vid_t u = 0, v = 0;
    for (int i = 0; i < 32; ++i) {
        float p = curand_uniform(&rng);
        if (p < a) {}
        else if (p < a + b) { v |= (1u << i); }
        else if (p < a + b + c) { u |= (1u << i); }
        else { u |= (1u << i); v |= (1u << i); }
    }
    u = u % vertices;
    v = v % vertices;
    edges[tid] = make_edge(u, v);

    float w = (curand_uniform(&rng) * MAX_WEIGHT);
    weights[tid] = w < 1.0f ? 1.0f : w;
}

// =========== Dynamic Edge Update (example logic) ===========

__global__ void simple_csr_from_edges(
    Edge* edges, real_t* weights, size_t n_edges,
    size_t* xadj, vid_t* adj, real_t* weight,
    vid_t vertices)
{
    // 简化处理，仅限用于演示，假设边已按源点排序
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_edges) return;

    Edge e = edges[tid];
    adj[tid] = e.v;
    weight[tid] = weights[tid];

    // 用atomic加边计数
    atomicAdd(&xadj[e.u + 1], 1);
}

__global__ void finalize_xadj(size_t* xadj, vid_t vertices) {
    for (int i = 1; i <= vertices; ++i) {
        xadj[i] += xadj[i - 1];
    }
}

// ========== Main Logic ==========

int main() {
    const size_t V = 1 << 16;
    const size_t E = 1000000;

    gpu_graph g;
    g.allocate(V, E);

    Edge* d_new_edges;
    real_t* d_new_weights;

    cudaMalloc(&d_new_edges, sizeof(Edge) * E);
    cudaMalloc(&d_new_weights, sizeof(real_t) * E);

    // 1. Generate new edges on GPU using RMAT
    generate_rmat_edges_gpu<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_new_edges, d_new_weights, E, V,
        0.5, 0.2, 0.1, time(NULL));

    cudaDeviceSynchronize();

    // 2. Rebuild CSR structure from these edges (simple version)
    g.free_all();
    g.allocate(V, E);

    cudaMemset(g.d_xadj, 0, sizeof(size_t) * (V + 1));

    simple_csr_from_edges<<<(E + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_new_edges, d_new_weights, E,
        g.d_xadj, g.d_adj, g.d_weight, V);

    finalize_xadj<<<1, 1>>>(g.d_xadj, V);

    cudaDeviceSynchronize();

    std::cout << "Graph with " << E << " edges generated and loaded on GPU (CSR)." << std::endl;

    // Cleanup
    g.free_all();
    cudaFree(d_new_edges);
    cudaFree(d_new_weights);
    return 0;
}
