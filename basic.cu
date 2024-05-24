#include "framework.cuh"

/**
 * @brief Print all CUDA support device.
 * 
 */
void framework::getDeviceName()
{
	printf("CUDA Device Info\n");

	int deviceCount = 0;
	cudaError_t error_id;

	error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("Error cudaGetDeviceCount() : %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		exit(EXIT_FAILURE);
	}
	else 
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);

	cudaDeviceProp deviceProp;

	// Iterate through all the devices found
	for (int i = 0; i < deviceCount; ++i) {
		cudaSetDevice(i);
		cudaGetDeviceProperties(&deviceProp, i);
		printf("Device: %d, %s\n\n", i, deviceProp.name);
	}
}

void framework::getDeviceInfo(size_t deviceID)
{
	cudaSetDevice(deviceID);

	cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, deviceID);
       printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
       printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
       printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
       printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
       printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
       printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
       printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
       printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
       printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
       printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);
       printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);
       printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
       printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
       printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
       printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
       printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
       printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
       printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
       printf("  Device supports Managed Memory:                %s\n",
           deviceProp.managedMemory ? "Yes" : "No");
       printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
       printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
       printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
       printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
}


void framework::bufferedImageLoader(cv::Mat& image, float* data) 
{
    cv::resize(image, image, cv::Size(224, 224));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    cv::Mat If32;
    image.convertTo(If32, CV_32FC3);
    
    // OpenCV HWC -> CHW
    #pragma omp parallel for
    for (int c = 0; c < image.channels(); ++c) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                data[c * image.rows * image.cols + y * image.cols + x] = static_cast<float>(If32.at<cv::Vec3f>(y, x)[c] / 255.0f);
            }
        }
    }
#ifdef __DEBUG__
    printf("c : %d, cols : %d, rows : %d\n", image.channels(), image.cols, image.rows);

    for (int i = 0; i < 256; ++i)
        printf("%f, ", data[i]);

    printf("\n");
#endif
}


inline float framework::IoU(const bbox_t &bbox_1, const bbox_t &bbox_2) 
{
    bbox_t bbox_and {0, 0.0f, std::max(bbox_1.x_tl, bbox_2.x_tl), std::max(bbox_1.y_tl, bbox_2.y_tl), std::min(bbox_1.x_br, bbox_2.x_br), std::min(bbox_1.y_br, bbox_2.y_br)};

    float bbox_1_area = std::abs(bbox_1.x_tl - bbox_1.x_br) * std::abs(bbox_1.y_tl - bbox_1.y_br);
    float bbox_2_area = std::abs(bbox_2.x_tl - bbox_2.x_br) * std::abs(bbox_2.y_tl - bbox_2.y_br);
    float bbox_and_area = std::max(bbox_and.x_br - bbox_and.x_tl, 0.0f) * std::max(bbox_and.y_br - bbox_and.y_tl, 0.0f);
    float iou = bbox_and_area / (bbox_1_area + bbox_2_area - bbox_and_area + 1e-6f);
#ifdef __DEBUG__
    printf("%f\n", bbox_1_area);
    printf("%f\n", bbox_2_area);
    printf("%f\n", bbox_and_area);
    printf("%f\n", iou);
#endif

    return iou;
}


void framework::NMS(std::vector<bbox_t> &bboxes, const float iouThreshold, const float probThreshold) 
{
    // Reduce over probability
    bboxes.erase(std::remove_if(bboxes.begin(), bboxes.end(), [&](bbox_t a) { return a.confidence < probThreshold; }), bboxes.end());
    
    // Sort over confidence in descending order
    std::sort(bboxes.begin(), bboxes.end(), [&](bbox_t a, bbox_t b) { return a.confidence > b.confidence; });
    
    // Reduce over iouThreshold
    float iou = 0.0f;
    for (int i = 0; i < bboxes.size(); ++i) 
    {
        for (int j = i + 1; j < bboxes.size(); ++j)
        {
            if (bboxes[i].objClass != bboxes[j].objClass)
                continue;
            iou = IoU(bboxes[i], bboxes[j]);
#ifdef __DEBUG__
            printf("%d, %d, %f", bboxes[i].objClass, bboxes[j].objClass, iou);
#endif
            if (iou > iouThreshold) {
#ifdef __DEBUG__
                printf(" -> delete\n");
#endif
                bboxes.erase(bboxes.begin() + j);
                j--;
            }
#ifdef __DEBUG__
            else
                printf("\n");
#endif
        }
    }
}
