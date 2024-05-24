#include "framework.cuh"

__global__ void DWConv2D_cout(int inputH, int inputW, int cin, int outputH, int outputW, int cout, int batchSize, int stride, int kSize, int pad, float* X, float* W, float* B, float* Y)
{
    unsigned int x_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_global_idx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z_global_idx = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef __KERNEL_DEBUG__
    if ((x_global_idx + y_global_idx + z_global_idx) == 0) {
        unsigned int x_global_size = gridDim.x * blockDim.x;
        unsigned int y_global_size = gridDim.y * blockDim.y;
        unsigned int z_global_size = gridDim.z * blockDim.z;
        
        printf("Grid Size : [%u, %u, %u]\n", gridDim.x, gridDim.y, gridDim.z);
        printf("Block Size : [%u, %u, %u]\n", blockDim.x, blockDim.y, blockDim.z);
        printf("Global Size : [%u, %u, %u]\n", x_global_size, y_global_size, z_global_size);
        printf("Input Height : %d\n", inputH);
        printf("Input Width : %d\n", inputW);
        printf("Input Channel : %d\n", cin);
        printf("Output Height : %d\n", outputH);
        printf("Output Width : %d\n", outputW);
        printf("Output Channel : %d\n", cout);
        printf("Batch Size : %d\n", batchSize);
        printf("Stride : %d\n", stride);
        printf("Kernel Size : %d\n", kSize);
        printf("Padding : %d\n", pad);
    }
#endif

    int width  = x_global_idx;
    int height = y_global_idx;
    int cOut   = z_global_idx;

    int wSubSize = kSize * kSize;
    int xOneBatchSize = inputH * inputW * cin;
    int xMapSize = inputH * inputW;
    int yOneBatchSize = outputH * outputW * cout;
    int yMapSize = outputH * outputW;

    float  sum;
    int indexW;
	for(int N = 0; N < batchSize; ++N)
	{
		sum    = 0;
		indexW = 0;
		for(int kh = 0; kh < kSize; ++kh)
		{
			for(int kw = 0; kw < kSize; ++kw, ++indexW)
			{
				int hp = height * stride + kh - pad;
				int wp = width  * stride + kw - pad;
				if(hp >= 0 && wp >=0 && hp < inputH && wp < inputW)
					sum += W[cOut * wSubSize + indexW] * X[N * xOneBatchSize + cOut * xMapSize + hp * inputW + wp];
			}
		}
        if(B != nullptr)
            Y[N * yOneBatchSize + cOut * yMapSize + height * outputW + width] = sum + B[cOut];
        else
            Y[N * yOneBatchSize + cOut * yMapSize + height * outputW + width] = sum;
	}
}


__global__ void DWConv2D_shared(int inputH, int inputW, int cin, int outputH, int outputW, int cout, int batchSize, int stride, int kSize, int pad, float* X, float* W, float* B, float* Y)
{
    unsigned int x_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_global_idx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z_global_idx = blockIdx.z * blockDim.z + threadIdx.z;

#ifdef __KERNEL_DEBUG__
    if ((x_global_idx + y_global_idx + z_global_idx) == 0) {
        unsigned int x_global_size = gridDim.x * blockDim.x;
        unsigned int y_global_size = gridDim.y * blockDim.y;
        unsigned int z_global_size = gridDim.z * blockDim.z;
        
        printf("Grid Size : [%u, %u, %u]\n", gridDim.x, gridDim.y, gridDim.z);
        printf("Block Size : [%u, %u, %u]\n", blockDim.x, blockDim.y, blockDim.z);
        printf("Global Size : [%u, %u, %u]\n", x_global_size, y_global_size, z_global_size);
        printf("Input Height : %d\n", inputH);
        printf("Input Width : %d\n", inputW);
        printf("Input Channel : %d\n", cin);
        printf("Output Height : %d\n", outputH);
        printf("Output Width : %d\n", outputW);
        printf("Output Channel : %d\n", cout);
        printf("Batch Size : %d\n", batchSize);
        printf("Stride : %d\n", stride);
        printf("Kernel Size : %d\n", kSize);
        printf("Padding : %d\n", pad);
    }
#endif

    int width  = x_global_idx;
    int height = y_global_idx;
    int cOut   = z_global_idx;

    int wSubSize = kSize * kSize;
    int xOneBatchSize = inputH * inputW * cin;
    int xMapSize = inputH * inputW;
    int yOneBatchSize = outputH * outputW * cout;
    int yMapSize = outputH * outputW;

    extern __shared__ float sharedWeight[];

    // Setup shared memory
    int indexW = 0;
#ifdef DEBUG_OLD_SHARE
    for (; indexW < wSubSize; ++indexW)
        sharedWeight[threadIdx.z * wSubSize + indexW] = W[cOut * wSubSize + indexW];
#endif

    int sharedNum = wSubSize * blockDim.z;
    int threadLocalIdx = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int threadLocalDim = blockDim.x * blockDim.y * blockDim.z;

    /*
    for (int i = 0; i <= sharedNum / threadLocalDim; ++i) {
        if ((i * threadLocalDim + threadLocalIdx) < sharedNum)
            sharedWeight[i * threadLocalDim + threadLocalIdx] = W[(sharedNum * blockIdx.z) + (i * threadLocalDim) + threadLocalIdx];
    }
    */

    for (int i = 0; i < sharedNum; i += threadLocalDim) {
        if ((i + threadLocalIdx) < sharedNum)
            sharedWeight[i + threadLocalIdx] = W[(sharedNum * blockIdx.z) + i + threadLocalIdx];
    }

    __syncthreads();

    float sum;
	for(int N = 0; N < batchSize; ++N)
	{
		sum    = 0;
		indexW = 0;
		for(int kh = 0; kh < kSize; ++kh)
		{
			for(int kw = 0; kw < kSize; ++kw, ++indexW)
			{
				int hp = height * stride + kh - pad;
				int wp = width  * stride + kw - pad;
				if(hp >= 0 && wp >=0 && hp < inputH && wp < inputW)
					sum += sharedWeight[threadIdx.z * wSubSize + indexW] * X[N * xOneBatchSize + cOut * xMapSize + hp * inputW + wp];
			}
		}
        if(B != nullptr)
            Y[N * yOneBatchSize + cOut * yMapSize + height * outputW + width] = sum + B[cOut];
        else
            Y[N * yOneBatchSize + cOut * yMapSize + height * outputW + width] = sum;
	}
}


framework::dwconv2d::dwconv2d(layer* p, int ks, int stride, int pad, bool use_bias, uint8_t version, dim3 blockSize)
{
    name = "DW-Conv2D";
    useBias = use_bias;
    padSize = pad;
    this->stride = stride>2?2:stride;
    kSize = ks;
    prvLayer = p;
    prvShape = prvLayer->getShape();
    shape.N = prvShape.N;
    shape.C = prvShape.C;
    shape.H = (prvShape.H - kSize + 2 * padSize) / stride + 1;
    shape.W = (prvShape.W - kSize + 2 * padSize) / stride + 1;
    memSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);
    
    // Select compute version (NoShared : 0, WithShared : 1)
    this->version = version;

    cudaError_t error_id;
    
    error_id = cudaMalloc(&gMEM, memSize);
	if (error_id != cudaSuccess) {
		printf("Error %s cudaMalloc() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

    error_id = cudaDeviceSynchronize();
    if (error_id != cudaSuccess) {
        printf("Error %s cudaDeviceSynchronize() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    weightSize = shape.C * kSize * kSize;
    biasSize = shape.C;
    setNParam(weightSize + int(useBias) * biasSize);
    outputSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);

    dimBlock.x = blockSize.x;
    dimBlock.y = blockSize.y;
    dimBlock.z = blockSize.z;

    dimGrid.x = (shape.W + blockSize.x - 1) / blockSize.x;
    dimGrid.y = (shape.H + blockSize.y - 1) / blockSize.y;
    dimGrid.z = (shape.C + blockSize.z - 1) / blockSize.z;

    jobSize.x = shape.W;
    jobSize.y = shape.H;
    jobSize.z = shape.C;
#ifdef __DEBUG__
    printf("%s\nGlobal Work Size [%u, %u, %u]\nGPU Memory Size : %ld\n", name.c_str(), (dimGrid.x * dimBlock.x), (dimGrid.y * dimBlock.y), (dimGrid.z * dimBlock.z), memSize);
    printf("\nBlockDim : [%u, %u, %u]\nGridDim : [%u, %u, %u]\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
#endif
}

/**
 * @brief Launch Kernel with Hyperparameter set in constructor.
 * 
 */
void framework::dwconv2d::run()
{
    cudaError_t error_id;
    float* prvMEM = prvLayer->getGPUMem();

    switch (version) {
        case 0:
	        DWConv2D_cout<<<dimGrid, dimBlock>>>(prvShape.H, prvShape.W, prvShape.C, shape.H, shape.W, shape.C, shape.N, stride, kSize, padSize, prvMEM, wMEM, bMEM, gMEM);
            break;
        case 1:
            int sharedMem = kSize * kSize * dimBlock.z * sizeof(float);
#ifdef __DEBUG__
            printf("%s Shared Memory Size : %d\n", name.c_str(), sharedMem);
#endif
            DWConv2D_shared<<<dimGrid, dimBlock, sharedMem>>>(prvShape.H, prvShape.W, prvShape.C, shape.H, shape.W, shape.C, shape.N, stride, kSize, padSize, prvMEM, wMEM, bMEM, gMEM);
            break;
    }
	cudaDeviceSynchronize();
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
		printf("Error: %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}

/**
 * @brief Set layer weight & bias parameter
 * 
 * @param buffer CPU buffer to initial weight & bias parameter
 */
size_t framework::dwconv2d::setParam(float* buffer)
{
  	cudaError_t error_id;

    error_id = cudaMalloc(&wMEM, weightSize * sizeof(float));
	if (error_id != cudaSuccess) {
		printf("Error %s cudaMalloc() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	error_id = cudaMemcpy(wMEM, buffer, weightSize * sizeof(float), cudaMemcpyHostToDevice);
	if (error_id != cudaSuccess) {
		printf("Error %s cudaMemcpy() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	error_id = cudaDeviceSynchronize();
	if (error_id != cudaSuccess) {
		printf("Error %s cudaDeviceSynchronize() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
    
    if (useBias) {
        error_id = cudaMalloc(&bMEM, biasSize * sizeof(float));
	    if (error_id != cudaSuccess) {
	    	printf("Error %s cudaMalloc() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
	    	exit(EXIT_FAILURE);
	    }

	    error_id = cudaMemcpy(bMEM, (buffer + weightSize), biasSize * sizeof(float), cudaMemcpyHostToDevice);
	    if (error_id != cudaSuccess) {
	    	printf("Error %s cudaMemcpy() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
	    	exit(EXIT_FAILURE);
	    }

	    error_id = cudaDeviceSynchronize();
	    if (error_id != cudaSuccess) {
	    	printf("Error %s cudaDeviceSynchronize() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
	    	exit(EXIT_FAILURE);
	    }
    }
    return weightSize + int(useBias) * biasSize;
}
