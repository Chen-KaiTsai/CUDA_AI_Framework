#include "framework.cuh"

__global__ void ReLU6(float* X, float* Y)
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
    }
#endif

    int i = x_global_idx;
    Y[i] = min(max(0.0f, X[i]), 6.0f);
}

framework::relu6::relu6(layer* p, bool inplace_, dim3 blockSize)
{
    name = "ReLU6";
    prvLayer = p;
    shape = prvLayer->getShape();

    if(inplace_) {
        gMEM = prvLayer->getGPUMem();
        memSize = 0;
    }
    else {
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

    setNParam(0);
    outputSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);

    dimBlock.x = blockSize.x;
    
    dimGrid.x = (shape.N * shape.C * shape.H * shape.W + blockSize.x - 1) / blockSize.x;

    jobSize.x = shape.N * shape.C * shape.H * shape.W;
#ifdef __DEBUG__
    printf("%s\nGlobal Work Size [%u, %u, %u]\nGPU Memory Size : %ld\n", name.c_str(), (dimGrid.x * dimBlock.x), (dimGrid.y * dimBlock.y), (dimGrid.z * dimBlock.z), memSize);
    printf("\nBlockDim : [%u, %u, %u]\nGridDim : [%u, %u, %u]\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
#endif
}

#ifdef USE_CUDNN_ACTIVATION
void framework::relu6::run()
{
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    cudnnTensorDescriptor_t prvMEM_desc;
    cudnnTensorDescriptor_t gMEM_desc;

    cudnnCreateTensorDescriptor(&prvMEM_desc);
    cudnnCreateTensorDescriptor(&gMEM_desc);

    cudnnSetTensor4dDescriptor(prvMEM_desc, format, dtype, shape.N, shape.C, shape.H, shape.W);
    cudnnSetTensor4dDescriptor(gMEM_desc, format, dtype, shape.N, shape.C, shape.H, shape.W);

    cudnnActivationDescriptor_t relu6_desc;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_CLIPPED_RELU;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&relu6_desc);
    cudnnSetActivationDescriptor(relu6_desc, mode, prop, 6.0f);

    float alpha[1] = {1.0f};
    float beta[1] = {0.0f};

    float* prvMEM = prvLayer->getGPUMem();
    cudnnActivationForward(handle, relu6_desc, alpha, prvMEM_desc, prvMEM, beta, gMEM_desc, gMEM);

    cudaDeviceSynchronize();
    cudnnDestroy(handle);
}
#else
void framework::relu6::run()
{
    cudaError_t error_id;
    float* prvMEM = prvLayer->getGPUMem();

	ReLU6<<<dimGrid, dimBlock>>>(prvMEM, gMEM);
	cudaDeviceSynchronize();
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
		printf("Error: %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}
#endif

size_t framework::relu6::setParam(float* buffer) { return 0; }
