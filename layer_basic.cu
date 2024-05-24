#include "framework.cuh"

/**
 * @brief Copy current guest(GPU) memory to host(CPU)
 * 
 * @todo the outputSize should be stored (differ from OpenCL version)
 */
void framework::layer::detach()
{
#ifdef __DEBUG__
    printf("%s OutputSize : %ld\n", name.c_str(), outputSize);
#endif
    if(cMEM == nullptr)
        cMEM = new float[outputSize / sizeof(float)];

    cudaError_t error_id;

	error_id = cudaMemcpy(cMEM, gMEM, outputSize, cudaMemcpyDeviceToHost);
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

