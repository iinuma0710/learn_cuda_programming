#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_functions.h> // ベンチマーク計測のため

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16


////////////////////////////////////////////////////////////////////////////////
//! GPU で次のような計算をする
//! C = alpha * A * B + beta * C
//! @param A          デバイスで計算に使う行列 A
//! @param B          デバイスで計算に使う行列 B
//! @param C          デバイスで計算に使う行列 C
//! @param N          行列 A と C の高さ
//! @param M          行列 B と C の幅
//! @param K          行列 A の幅，行列 B の 高さ
//! @param alpha      行列の掛け算のときに掛けるスカラ値
//! @param beta       行列の和を取るときに掛けるスカラ値
////////////////////////////////////////////////////////////////////////////////

__global__ void sgemm_gpu_kernel(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * K + col];
    }

    C[row * M + col] = alpha * sum + beta * C[row * M + col];  
}

void sgemm_gpu(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
    sgemm_gpu_kernel<<<dimGrid, dimBlock>>>(A, B, C, N, M, K, alpha, beta);
}

void random_init(float *data, int size)
{
	for (int i = 0; i < size; ++i) {
		data[i] = (rand() & 0xFF) / (float)RAND_MAX;
	}
}

void performance_estimation(void(*sgemm)(const float *, const float *, float *, int, int, int, float, float),
    const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    int test_iterations = 100;

	// タイマの作成
	StopWatchInterface *timer = 0;

	// ウォームスタートで計算を開始する
	sgemm(A, B, C, N, M, K, alpha, beta);

	// イベントの開始を記録する
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	// 操作本体
	for (int i = 0; i < test_iterations; i++) {
		sgemm(A, B, C, N, M, K, alpha, beta);
	}

	// GPU での操作が終わったら終了時刻を記録する
	sdkStopTimer(&timer);

	// パフォーマンスの計算と表示
	float operation_time = sdkGetAverageTimerValue(&timer);
	float operation_time_1_epoch = operation_time / test_iterations;
	printf("Operation Time= %.4f msec\n", operation_time_1_epoch);

	// タイマを破棄
	sdkDeleteTimer(&timer);
}

int main(void)
{
	float *A, *B, *C;
	float *d_A, *d_B, *d_C;
	int N, M, K;
	float alpha = 2.f;
	float beta = 1.f;
	N = M = K = 2048;

	// CPU 側で1次元のメモリ領域を確保する
	A = (float *)malloc(N * K * sizeof(float));
	B = (float *)malloc(K * M * sizeof(float));
	C = (float *)malloc(N * M * sizeof(float));

	// GPU 側で1次元のメモリ領域を確保する
	cudaMalloc((void **)&d_A, N * K * sizeof(float));
	cudaMalloc((void **)&d_B, K * M * sizeof(float));
	cudaMalloc((void **)&d_C, N * M * sizeof(float));

	// データの初期化
	random_init(A, N * K);
	random_init(B, K * M);
	random_init(C, N * M);

	// GPU 側にメモリをコピー
	cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, A, K * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, A, N * M * sizeof(float), cudaMemcpyHostToDevice);

	// パフォーマンスの計測
	//sgemm_gpu(d_A, d_B, d_C, N, M, K, alpha, beta);
	performance_estimation(sgemm_gpu, d_A, d_B, d_C, N, M, K, alpha, beta);

	// GPU 側のメモリ解放
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// CPU 側のメモリ解放
	free(A);
	free(B);
	free(C);

	return 0;
}