#include <iostream>
#include<stdio.h>
#include <math.h>

#define STRIDE_64K 65536

__global__ void init(int n, float *x, float *y) {
    int lane_id = threadIdx.x & 31;
    size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
    size_t warps_per_grid = (blockDim.x * gridDim.x) >> 5;
    size_t warp_total = ((sizeof(float)*n) + STRIDE_64K-1) / STRIDE_64K;

    for( ; warp_id < warp_total; warp_id += warps_per_grid) {
        #pragma unroll
        for(int rep = 0; rep < STRIDE_64K/sizeof(float)/32; rep++) {
            size_t ind = warp_id * STRIDE_64K/sizeof(float) + rep * 32 + lane_id;
            if (ind < n) {
              x[ind] = 1.0f;
              y[ind] = 2.0f;
            }
        }
    }
}
 
// 2つの配列を足し合わせる CUDA カーネル
__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
 
int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // Unified memory の割当 (CPU と GPU からアクセス可能)
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // GPU 上で 1M この要素からなるカーネルを立ち上げる
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    size_t warp_total = ((sizeof(float)*N) + STRIDE_64K-1) / STRIDE_64K;
    int numBlocksInit = (warp_total*32) / blockSize;
    
    init<<<numBlocksInit, blockSize>>>(N, x, y);
    add<<<numBlocks, blockSize>>>(N, x, y);

    // カーネルの実行が終わるまで待つ
    cudaDeviceSynchronize();

    // エラーがないかチェック (全て 3.0f になっているはず)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // メモリ解放
    cudaFree(x);
    cudaFree(y);

    return 0;
}
