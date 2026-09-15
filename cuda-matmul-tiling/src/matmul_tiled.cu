#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define K 1024
#define N 1024
#define TILE_SIZE 16

__global__ void matmulTiled(const float* A, const float* B, float* C) {
    // Shared memory tiles — visible to all threads in this block, reused many times
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Slide across K in chunks of TILE_SIZE — you calculated this: 1024/16 = 64 iterations
    for (int t = 0; t < K / TILE_SIZE; t++) {

        // Each thread loads exactly ONE element into shared memory (cooperative load, not redundant)
        tileA[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // Wait for ALL threads in the block to finish loading before anyone starts computing
        __syncthreads();

        // Now every thread reuses the shared tile many times — no redundant global memory hits
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Wait for everyone to finish computing before the next iteration overwrites the shared tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, M / TILE_SIZE);

    matmulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (h_C[i] != (float)K) { correct = false; break; }
    }
    printf(correct ? "SUCCESS: all values correct\n" : "FAILURE: mismatch found\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
