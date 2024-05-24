#include "framework.cuh"

__global__ void MaxPool2D(int inputH, int inputW, int cin, int outputH, int outputW, int cout, int batchSize, int stride, int kSize, int pad, float* X, float* W, float* B, float* Y)
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
        printf("Padding %d\n", pad);
    }
#endif

    int width  = x_global_idx;
    int height = y_global_idx;
    int cOut   = z_global_idx;

    //int wSubSize = kSize * kSize * cin;
    int xOneBatchSize = inputH * inputW * cin;
    int xMapSize = inputH * inputW;
    int yOneBatchSize = outputH * outputW * cout;
    int yMapSize = outputH * outputW;

    for (int N = 0; N < batchSize; ++N)
    {
        float max_value = FLOAT32MIN;

        for (int kh = 0; kh < kSize; ++kh)
        {
            for (int kw = 0; kw < kSize; ++kw)
            {
                int hp = height * stride + kh - pad;
                int wp = width  * stride + kw - pad;
                if (hp >= 0 && wp >= 0 && hp < inputH && wp < inputW)
                    max_value = max(max_value, X[N * xOneBatchSize + cOut *xMapSize + hp * inputW + wp]);
            }
        }

        Y[N * yOneBatchSize + cOut * yMapSize + height * cout + width] = max_value;
    }
}

framework::maxpool2d::maxpool2d(layer* p, int ks, int s, int pad, dim3 blockSize)
{
    name = "MaxPool2D";
    padSize = pad;
    stride = s>2?2:s;
    kSize = ks;
    prvLayer = p;
    shape_t prvShape = prvLayer->getShape();
    shape.N = prvShape.N;
    shape.C = prvShape.C;
    shape.H = (prvShape.H - kSize + 2 * padSize) / stride + 1;
    shape.W = (prvShape.W - kSize + 2 * padSize) / stride + 1;
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

    setNParam(0);
    outputSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);

    dimBlock.x = blockSize.x;
    dimBlock.y = blockSize.y;
    dimBlock.z = blockSize.z;

    dimGrid.x = (shape.W + blockSize.x - 1) / blockSize.x;
    dimGrid.y = (shape.H + blockSize.y - 1) / blockSize.y;
    dimGrid.z = (shape.C + blockSize.z - 1) / blockSize.z;

    jobSize.x = shape.W;
    jobSize.y = shape.H;
    jobSize.y = shape.C;
#ifdef __DEBUG__
    printf("%s\nGlobal Work Size [%u, %u, %u]\nGPU Memory Size : %ld\n", name.c_str(), (dimGrid.x * dimBlock.x), (dimGrid.y * dimBlock.y), (dimGrid.z * dimBlock.z), memSize);
    printf("\nBlockDim : [%u, %u, %u]\nGridDim : [%u, %u, %u]\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
#endif
}

void framework::maxpool2d::run()
{
    cudaError_t error_id;
    float* prvMEM = prvLayer->getGPUMem();

	MaxPool2D<<<dimGrid, dimBlock>>>(prvShape.H, prvShape.W, prvShape.C, shape.H, shape.W, shape.C, shape.N, stride, kSize, padSize, prvMEM, wMEM, bMEM, gMEM);
	cudaDeviceSynchronize();
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
		printf("Error: %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}

size_t framework::maxpool2d::setParam(float* buffer) { return 0; }
