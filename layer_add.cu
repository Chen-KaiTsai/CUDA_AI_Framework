#include "framework.cuh"

__global__ void Add(float* X, float* SC, float* Y)
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
    Y[i] = X[i] + SC[i];
}


framework::add::add(layer* p, layer* src, bool inplace_, dim3 blockSize)
{
    name = "Add";
    prvLayer = p;
    srcLayer = src;
    shape = prvLayer->getShape();
    shape_t prvShape = shape;
    shape_t srcShape = srcLayer->getShape();

    if (srcShape.H != prvShape.H || prvShape.W != srcShape.W || prvShape.C != srcShape.C || prvShape.N != srcShape.N)
    {
        printf("Skip Connection with feature map size not match\n");
        exit(EXIT_FAILURE);
    }

    if (inplace_) {
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
    printf("\nBlockDim : [%u, %u, %u]\nGridDim : [%u, %u, %u]\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
    printf("%s\nGlobal Work Size [%u, %u, %u]\nGPU Memory Size : %ld\n", name.c_str(), (dimGrid.x * dimBlock.x), (dimGrid.y * dimBlock.y), (dimGrid.z * dimBlock.z), memSize);
#endif
}

void framework::add::run()
{
    cudaError_t error_id;
    float* prvMEM = prvLayer->getGPUMem();
    float* srcMEM = srcLayer->getGPUMem();

	Add<<<dimGrid, dimBlock>>>(prvMEM, srcMEM, gMEM);
	cudaDeviceSynchronize();
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
		printf("Error: %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}

size_t framework::add::setParam(float* buffer) { return 0; }
