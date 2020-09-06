#include <stdio.h>
#include <stdlib.h>

#define N 2048
#define BLOCK_SIZE 32


__global__ void matrix_transpose_naive (int *input, int *output)
{
    int index_X = threadIdx.x + blockIdx.x * blockDim.x;
    int index_Y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = index_Y * N + index_X;
    int transposed_index = index_X * N + index_Y;

    // 非結合アクセスでストア
    output[transposed_index] = input[index];

    // 非結合アクセスでロード
    // output[index] = input[transposed_index];
}


__global__ void matrix_transpose_shared (int *input, int *output)
{
    __shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE];
    
    // 転置前のグローバルメモリ上のインデックス
    int index_X = threadIdx.x + blockIdx.x * blockDim.x;
    int index_Y = threadIdx.y + blockIdx.y * blockDim.y;

    // 転置後のグローバルメモリ上のインデックス
    int t_index_X = threadIdx.x + blockIdx.y * blockDim.x;
    int t_index_Y = threadIdx.y + blockIdx.x * blockDim.y;

    // ローカルのインデックス
    int local_index_X = threadIdx.x;
    int local_index_Y = threadIdx.y;

    int index = index_Y * N + index_X;
    int transposed_index = t_index_Y * N + t_index_X;

    // グローバルメモリから結合アクセスが可能なように共有メモリに読み出してから転置
	sharedMemory[local_index_X][local_index_Y] = input[index];

	__syncthreads();

	// 転置したデータをグローバルメモリに書き出し
	output[transposed_index] = sharedMemory[local_index_Y][local_index_X];
}

// インデックスと同じ数字で行列を埋める
void fill_array(int *data) {
	for(int idx = 0; idx < (N * N); idx++) {
        data[idx] = idx;
    }
}

// 結果を表示
void print_output(int *a, int *b) {
	printf("\n Original Matrix::\n");
	for(int idx=0 ; idx < (N * N); idx++) {
		if(idx % N == 0)
			printf("\n");
		printf(" %d ",  a[idx]);
	}
	printf("\n Transposed Matrix::\n");
	for(int idx = 0 ; idx < (N * N); idx++) {
		if(idx % N == 0)
			printf("\n");
		printf(" %d ",  b[idx]);
	}
}

int main(void)
{
    int *a, *b;
    int *d_a, *d_b;
    
    int size = N * N * sizeof(int);
    
    // ホスト側でメモリを確保して中身を書き込む
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    fill_array(a);

    // デバイス側のメモリを確保
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    // データをコピーする
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 転置の実行
    dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 gridSize(N/BLOCK_SIZE,N/BLOCK_SIZE,1);
    // matrix_transpose_naive<<<gridSize,blockSize>>>(d_a,d_b);
    matrix_transpose_shared<<<gridSize,blockSize>>>(d_a,d_b);

    // 結果をホスト側に書き戻して表示
    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
    print_output(a, b);
    
    // メモリの解放
    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}