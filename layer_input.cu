#include "framework.cuh"

framework::input::input(shape_t s)
{
    name = "Input";
    prvLayer = nullptr;
    shape = s;
    memSize = s.N * s.C * s.H * s.W * sizeof(float);

    cudaError error_id;
    
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
    outputSize = s.N * s.C * s.H * s.W * sizeof(float);
#ifdef __DEBUG__
    printf("%s\nGPU Memory Size : %ld\n", name.c_str(), memSize);
#endif
}

void framework::input::run()
{
    if (cMEM != nullptr) {
        cudaError_t error_id;

	    error_id = cudaMemcpy(gMEM, cMEM, memSize, cudaMemcpyHostToDevice);
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
}

size_t framework::input::setParam(float* buffer) { return 0; };
