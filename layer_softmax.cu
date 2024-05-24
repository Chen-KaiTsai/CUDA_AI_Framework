#include "framework.cuh"

__global__ void Softmax(float* X, float* Y)
{
    unsigned int x_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_global_idx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z_global_idx = 0;

#ifdef __KERNEL_DEBUG__
    if ((x_global_idx + y_global_idx + z_global_idx) == 0) {
        unsigned int x_global_size = gridDim.x * blockDim.x;
        unsigned int y_global_size = gridDim.y * blockDim.y;
        unsigned int z_global_size = 0;
        
        printf("Grid Size : [%u, %u, %u]\n", gridDim.x, gridDim.y, gridDim.z);
        printf("Block Size : [%u, %u, %u]\n", blockDim.x, blockDim.y, blockDim.z);
        printf("Global Size : [%u, %u, %u]\n", x_global_size, y_global_size, z_global_size);
    }
#endif

    int node     = x_global_idx;
    int nodeSize = gridDim.x * blockDim.x;
    int batch    = y_global_idx;
    
    float sum = 0;
    for (int n = 0; n < nodeSize; ++n)
        sum += exp(X[batch * nodeSize + n]);
    Y[batch * nodeSize + node] = exp(X[batch * nodeSize + node]) / sum;
}

framework::softmax::softmax(layer* p, bool inplace_, dim3 blockSize)
{
    name =  "Softmax";
    prvLayer = p;
    shape = prvLayer->getShape();
    memSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);

    if(inplace_) {
        gMEM = prvLayer->getGPUMem();
        memSize = 0;
    }
    else {
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
    dimBlock.y = blockSize.y;
    
    //dimBlock.z = blockSize.z; // for optimized version
    
    dimGrid.x = (shape.C * shape.H * shape.W + blockSize.x - 1) / blockSize.x;
    dimGrid.y = (shape.N + blockSize.y - 1) / blockSize.y;

    //dimGrid.z = (inNodeSize + blockSize.y - 1) / blockSize.y; // for optimized version

    jobSize.x = shape.C * shape.H * shape.W;
    jobSize.y = shape.N;
#ifdef __DEBUG__
    printf("%s\nGlobal Work Size [%u, %u, %u]\nGPU Memory Size : %ld\n", name.c_str(), (dimGrid.x * dimBlock.x), (dimGrid.y * dimBlock.y), (dimGrid.z * dimBlock.z), memSize);
    printf("\nBlockDim : [%u, %u, %u]\nGridDim : [%u, %u, %u]\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
#endif    
}

/**
 * @brief Launch Kernel with Hyperparameter set in constructor.
 * 
 */
void framework::softmax::run()
{
    cudaError_t error_id;
    float* prvMEM = prvLayer->getGPUMem();

	Softmax<<<dimGrid, dimBlock>>>(prvMEM, gMEM);
	cudaDeviceSynchronize();
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
		printf("Error: %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}

size_t framework::softmax::setParam(float* buffer) { return 0; }
