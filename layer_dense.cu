#include "framework.cuh"

__global__ void Dense(int inputH, int inputW, int inNodeSize, int outputH, int outputW, int nodeSize, int batchSize, float* X, float* W, float* B, float* Y)
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
        printf("Input Height : %d\n", inputH);
        printf("Input Width : %d\n", inputW);
        printf("Input Node Size : %d\n", inNodeSize);
        printf("Output Height : %d\n", outputH);
        printf("Output Width : %d\n", outputW);
        printf("Layer Node Size : %d\n", nodeSize);
        printf("Batch Size : %d\n", batchSize);
    }
#endif

    int node  = x_global_idx;
    int batch = y_global_idx;
    
    float sum = 0;
    for (int nIn = 0; nIn < inNodeSize; ++nIn) // can be another dimension
        sum += X[batch * inNodeSize + nIn] * W[node * inNodeSize + nIn];
    if (B != nullptr)
        Y[batch * nodeSize + node] = sum + B[node];
    else
        Y[batch * nodeSize + node] = sum;
}

/**
 * @brief Limitation : SharedMemory size should be inNode Size to store prvNode*W (each thread within the block store to the SharedMem)
 * blockSize should equal to the size of node (dense layer node size).
 * We can have :
 * Step1 : prvNode * W (each thread)
 * Step2 : sum reduction on sharedMem on node and + bias (each block with one thread)
*/
__global__ void Dense_opt(int inputH, int inputW, int inNodeSize, int outputH, int outputW, int nodeSize, int batchSize, float* X, float* W, float* B, float* Y)
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
        printf("Input Node Size : %d\n", inNodeSize);
        printf("Output Height : %d\n", outputH);
        printf("Output Width : %d\n", outputW);
        printf("Layer Node Size : %d\n", nodeSize);
        printf("Batch Size : %d\n", batchSize);
    }
#endif

    int prvNode = x_global_idx;
    int node    = y_global_idx;
    int batch   = z_global_idx;
    
    extern __shared__ float sharedNodeTemp[];

    sharedNodeTemp[threadIdx.x] += X[batch * inNodeSize + prvNode] * W[node * inNodeSize + prvNode];

    __syncthreads();

    if(prvNode == 0)
    {
        // Sum Reduction on SharedMem for node
        float sum = 0;
        for (int i = 0; i < inNodeSize; ++i) {
            sum += sharedNodeTemp[i];
        }

        // Add bias and store the result
        if (B != nullptr)
            Y[batch * nodeSize + node] = sum + B[node];
        else
            Y[batch * nodeSize + node] = sum;
    }
}

framework::dense::dense(layer* p, int nodeSize, bool use_bias, dim3 blockSize)
{
    name =  "Dense";
    useBias = use_bias;
    prvLayer = p;
    shape_t prvShape = prvLayer->getShape();
    inNodeSize = prvShape.H * prvShape.W * prvShape.C;
    shape.N = prvShape.N;
    shape.C = nodeSize;
    shape.H = 1;
    shape.W = 1;
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

    weightSize = nodeSize * inNodeSize;
    biasSize = shape.C;
    setNParam(weightSize + int(useBias) * biasSize);
    outputSize = shape.N * shape.C * shape.H * shape.W * sizeof(float);

    dimBlock.x = blockSize.x;
    dimBlock.y = blockSize.y;
    
    //dimBlock.z = blockSize.z; // for optimized version
    
    dimGrid.x = (shape.C + blockSize.x - 1) / blockSize.x;
    dimGrid.y = (shape.N + blockSize.y - 1) / blockSize.y;

    //dimGrid.z = (inNodeSize + blockSize.y - 1) / blockSize.y; // for optimized version

    jobSize.x = shape.C;
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
void framework::dense::run()
{
    cudaError_t error_id;
    float* prvMEM = prvLayer->getGPUMem();

	Dense<<<dimGrid, dimBlock>>>(prvShape.H, prvShape.W, inNodeSize, shape.H, shape.W, shape.C, shape.N, prvMEM, wMEM, bMEM, gMEM);
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
size_t framework::dense::setParam(float* buffer)
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
