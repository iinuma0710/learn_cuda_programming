#include <iostream>
#include <math.h>

__global__ void init(int n, float *x, float *y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        x[i] = 1.0f;
        y[i] = 2.0f;
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
    init<<<numBlocks, blockSize>>>(N, x, y);
    add<<<numBlocks, blockSize>>>(N, x, y);

    // カーネルの実行が終わるまで待つ
    cudaDeviceSynchronize();

    // エラーがないかチェック (全て 3.0f になっているはず)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // メモリの解放
    cudaFree(x);
    cudaFree(y);

    return 0;
}
