#include "framework.cuh"

__global__ void BatchNorm(int outputH, int outputW, int cout, int batchSize, float epsilon, float* X, float* P, float* Y)
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
        printf("Output Height : %d\n", outputH);
        printf("Output Width : %d\n", outputW);
        printf("Output Channel : %d\n", cout);
        printf("Batch Size : %d\n", batchSize);
        printf("Epsilon : %f\n", epsilon);
    }
#endif

    int width  = x_global_idx;
    int height = y_global_idx;
    int cOut   = z_global_idx;

    int OneBatchSize = outputH * outputW * cout;
    int MapSize = outputH * outputW;

    for (int N = 0; N < batchSize; ++N)
    {
        float gamma = P[0 * cout + cOut];
        float beta  = P[1 * cout + cOut];
        float mean  = P[2 * cout + cOut];
        float var   = P[3 * cout + cOut];

        Y[N * OneBatchSize + cOut * MapSize + height * outputW + width] = gamma * ((X[N * OneBatchSize + cOut * MapSize + height * outputW + width] - mean) / sqrtf(var + 1e-5)) + beta;
    }
}

framework::batchnorm::batchnorm(layer* p, float epsilon_, bool inplace_, dim3 blockSize)
{
    name = "BatchNorm";
    prvLayer = p;
    shape = prvLayer->getShape();
    epsilon = epsilon_;
    paramSize = shape.C * 4;
    if(inplace_) {
        gMEM = prvLayer->getGPUMem();
        memSize = 0;
    }
    else
    {
        memSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);
        
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
    }
    setNParam(paramSize);
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

void framework::batchnorm::run()
{
	cudaError_t error_id;
    float* prvMEM = prvLayer->getGPUMem();

	BatchNorm<<<dimGrid, dimBlock>>>(shape.H, shape.W, shape.C, shape.N, epsilon, prvMEM, pMEM, gMEM);
	cudaDeviceSynchronize();
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
		printf("Error: %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}

size_t framework::batchnorm::setParam(float* buffer)
{
    cudaError_t error_id;

    error_id = cudaMalloc(&pMEM, paramSize * sizeof(float));
	if (error_id != cudaSuccess) {
		printf("Error %s cudaMalloc() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	error_id = cudaMemcpy(pMEM, buffer, paramSize * sizeof(float), cudaMemcpyHostToDevice);
	if (error_id != cudaSuccess) {
		printf("Error %s cudaMemcpy() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	error_id = cudaDeviceSynchronize();
	if (error_id != cudaSuccess) {
		printf("Error %s cudaDeviceSynchronize() : %d\n%s\n\n", name.c_str(), static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
    
    return paramSize;
}

