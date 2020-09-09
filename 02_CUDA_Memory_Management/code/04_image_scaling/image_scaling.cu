#include<stdio.h>
#include"scrImagePgmPpmPackage.h"

//Step 1: テクスチャメモリを宣言する
texture<unsigned char, 2, cudaReadModeElementType> tex;


// 画像をリサイズするカーネル
__global__ void createResizedImage(unsigned char *imageScaledData, int scaled_width, float scale_factor, cudaTextureObject_t texObj)
{
        const unsigned int tidX = blockIdx.x*blockDim.x + threadIdx.x;
        const unsigned int tidY = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned index = tidY*scaled_width+tidX;
       	
	// Step 3: CUDA カーネルのテクスチャ参照からテクスチャメモリを読み込む
	imageScaledData[index] = tex2D<unsigned char>(texObj,(float)(tidX*scale_factor),(float)(tidY*scale_factor));
}


int main(int argc, char*argv[])
{
	int height=0, width =0, scaled_height=0,scaled_width=0;
	float scaling_ratio=0.5;	// 画像の倍率を設定
	unsigned char*data;
	unsigned char*scaled_data,*d_scaled_data;

	char inputStr[1024] = {"aerosmith-double.pgm"};
	char outputStr[1024] = {"aerosmith-double-scaled.pgm"};
	cudaError_t returnValue;

	// テクスチャにリンクしている間使われる Channel Description の作成
	cudaArray* cu_array;
	cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, kind);

	get_PgmPpmParams(inputStr, &height, &width);	// 画像の高さや幅を取得
	data = (unsigned char*)malloc(height*width*sizeof(unsigned char));
	printf("\n Reading image width height and width [%d][%d]", height, width);
	scr_read_pgm(inputStr , data, height, width);	// 入力画像を読み込む

	scaled_height = (int)(height*scaling_ratio);
	scaled_width = (int)(width*scaling_ratio);
	scaled_data = (unsigned char*)malloc(scaled_height*scaled_width*sizeof(unsigned char));
	printf("\n scaled image width height and width [%d][%d]", scaled_height, scaled_width);

	// デバイス側のメモリを割り当てる
 	returnValue = cudaMallocArray( &cu_array, &channelDesc, width, height);
	returnValue = (cudaError_t)(returnValue | cudaMemcpyToArray( cu_array, 0, 0, data, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice));
	if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API Array Copy");

	// テクスチャの特定
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cu_array;

	// Step 2. テクスチャオブジェクトのパラメータを特定する
	struct cudaTextureDesc texDesc;
	// メモリにゼロをセットする
	memset(&texDesc, 0, sizeof(texDesc));
	// x 方向の次元のアドレスモードを Clamp に設定
	texDesc.addressMode[0] = cudaAddressModeClamp;
	// y 方向の次元のアドレスモードを Clamp に設定
	texDesc.addressMode[1] = cudaAddressModeClamp;
	// フィルタモードを Point に設定
	texDesc.filterMode = cudaFilterModePoint;
	// 要素のタイプを読込み，補間はしない
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// テクスチャオブジェクトの生成
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API Bind Texture");
	cudaMalloc(&d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char) );

    dim3 dimBlock(32, 32,1);
    dim3 dimGrid(scaled_width/dimBlock.x,scaled_height/dimBlock.y,1);
	printf("\n Launching grid with blocks [%d][%d] ", dimGrid.x,dimGrid.y);

    createResizedImage<<<dimGrid, dimBlock>>>(d_scaled_data, scaled_width, 1 / scaling_ratio, texObj);
	returnValue = (cudaError_t)(returnValue | cudaThreadSynchronize());
	returnValue = (cudaError_t)(returnValue | cudaMemcpy(scaled_data , d_scaled_data, scaled_height * scaled_width * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API kernel");

	// Step 4: テクスチャオブジェクトを破棄する
	cudaDestroyTextureObject(texObj);
	// detections と一緒に画像を保存
	scr_write_pgm( outputStr, scaled_data, scaled_height, scaled_width, "####" );
		
	if(data != NULL)
		free(data);
	if(cu_array !=NULL)
		cudaFreeArray(cu_array);
	if(scaled_data != NULL)
		free(scaled_data);
	if(d_scaled_data!=NULL)
		cudaFree(d_scaled_data);
	
	return 0;
}
