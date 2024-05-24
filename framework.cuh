/**
 * @author Chen-Kai Tsai
 * @brief first release
 * @version 1.0
 * @date 2023-02-15
 * 
 * @version 1.5 beta
 * @brief shared memory version (now only support for inference with batch=1)
 * @date 2023-03-12
 * 
 * @todo loop unrolling on convolutional layers
 * @todo vectorization on convolutional layers
 * 
 * @version 1.6 beta
 * @brief fix getArgumentReference reporting wrong jobSize
 * @date 2023-11-22
 * 
 */

#ifndef INCLUDE_FRAMEWORK_H_
#define INCLUDE_FRAMEWORK_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

#include <cstdio>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <limits>
#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// #define __DEBUG__
// #define __KERNEL_DEBUG__

#define EPSILON (1e-5)

const float FLOAT32MIN = std::numeric_limits<float>::min();

using shape_t = struct shape
{
    int N;
    int C;
    int H;
    int W;
};

using bbox_t = struct bbox 
{
    uint objClass;
    float confidence;
    float x_tl;
    float y_tl;
    float x_br;
    float y_br;
};

namespace framework /* basic.cu */
{
    void getDeviceName();
    void getDeviceInfo(size_t deviceID);
    inline float IoU(const bbox_t &bbox_1, const bbox_t &bbox_2);
    void NMS(std::vector<bbox_t> &bboxes, const float iouThreshold, const float probThreshold);
    void bufferedImageLoader(cv::Mat& image, float* data);
};


namespace framework
{
    /**
     * @brief Base class for all layers [ layer_basic.cu ]
     * 
     */
    class layer
    {
    protected:
        std::string name;
        float* cMEM = nullptr;
        float* gMEM = nullptr;
        layer* prvLayer;
        shape_t shape;
        shape_t prvShape;
        size_t outputSize = 0;
        size_t nParam = 0;
        size_t memSize = 0;
        dim3 dimGrid;
        dim3 dimBlock;
        dim3 jobSize;
        uint8_t version;

    public:
        layer() {}
        ~layer() {}
        virtual void run() = 0;
        virtual size_t setParam(float* buffer) = 0;
        // virtual void setKernel() = 0; // TODO might not need this
        
        void setCPUMem(float* m) { cMEM = m; };
        float* getCPUMem() { return cMEM; };
        
        void setGPUMem(float* m) { gMEM = m; };
        float* getGPUMem() { return gMEM; };

        size_t getGPUMemSize() { return memSize; };

        void setName(std::string n) { name = n; };
        std::string getName() { return name; };

        void setPrvLayer(layer* p) { prvLayer = p; };
        layer* getPrvLayer() { return prvLayer; };

        shape_t getShape() { return shape; };
        void setShape(shape_t s) { shape = s; };

        size_t getNParam() { return nParam; };
        void setNParam(size_t n) { nParam = n; };

        dim3 getGridSize() { return dimGrid; };
        dim3 getBlockSize() { return dimBlock; };
        dim3 getJobSize() { return jobSize; };
        
        void detach();
    };


    /**
     * @brief Input layer [ layer_input.cu ]
     * 
     * @param s input shape with shape_t type
     */
    class input : public layer
    {
    public:
        input() = delete;
        input(const input&) = delete;
        input(shape_t s);
        ~input() {};
        void run();
        size_t setParam(float* buffer);
    };


    class conv2d : public layer
    {
    protected:
        float* wMEM = nullptr;
        float* bMEM = nullptr;
        bool useBias;
        int padSize;
        int stride;
        int kSize;
        size_t weightSize;
        size_t biasSize;

    public:
        conv2d() = delete;
        conv2d(const conv2d&) = delete;
        /**
         * @brief 2D Convolution layer constructor [ layer_conv2d.cu ]
         * 
         * @param p Previous layer
         * @param cout Output feature map channel size
         * @param ks Kernel size
         * @param stride Stride size
         * @param pad Padding size
         * @param use_bias Use bias or not 
         */
        conv2d(layer* p, int cout, int ks, int stride = 1, int pad = 0, bool use_bias = true, uint8_t version = 0, dim3 blockSize = dim3(1, 1, 1));
        ~conv2d() {
            if (wMEM != nullptr)
                cudaFree(wMEM);
            if (bMEM != nullptr)
                cudaFree(bMEM);
            if (cMEM != nullptr)
                delete[] (float*)cMEM;
            if (gMEM != nullptr)
                cudaFree(gMEM);
            cudaDeviceSynchronize();
#ifdef __DEBUG__
            printf("%s delete\n", name.c_str());
#endif
        }
        void run();
        size_t setParam(float* buffer);
    };


    class dwconv2d : public layer
    {
    protected:
        float* wMEM = nullptr;
        float* bMEM = nullptr;
        bool useBias;
        int padSize;
        int stride;
        int kSize;
        size_t weightSize;
        size_t biasSize;

    public:
        dwconv2d() = delete;
        dwconv2d(const dwconv2d&) = delete;
        /**
         * @brief 2D DWConvolution layer constructor [ layer_dwconv2d.cu ]
         * 
         * @param p Previous layer
         * @param ks Kernel size
         * @param stride Stride size
         * @param pad Padding size
         * @param use_bias Use bias or not 
         */
        dwconv2d(layer* p, int ks, int stride = 1, int pad = 0, bool use_bias = true, uint8_t version = 0, dim3 blockSize = dim3(1, 1, 1));
        ~dwconv2d() {
            if (wMEM != nullptr)
                cudaFree(wMEM);
            if (bMEM != nullptr)
                cudaFree(bMEM);
            if (cMEM != nullptr)
                delete[] (float*)cMEM;
            if (gMEM != nullptr)
                cudaFree(gMEM);
            cudaDeviceSynchronize();
#ifdef __DEBUG__
            printf("%s delete\n", name.c_str());
#endif
        }
        void run();
        size_t setParam(float* buffer);
    };


    class pwconv2d : public layer
    {
    protected:
        float* wMEM = nullptr;
        float* bMEM = nullptr;
        bool useBias;
        int padSize;
        int stride;
        int kSize;
        size_t weightSize;
        size_t biasSize;

    public:
        pwconv2d() = delete;
        pwconv2d(const pwconv2d&) = delete;
        /**
         * @brief 2D PWConvolution layer constructor [ layer_dwconv2d.cu ]
         * 
         * @param p Previous layer
         * @param cout Output channel size
         * @param use_bias Use bias or not 
         */
        pwconv2d(layer* p, int cout, bool use_bias = true, uint8_t version = 0, dim3 blockSize = dim3(1, 1, 1));
        ~pwconv2d() {
            if (wMEM != nullptr)
                cudaFree(wMEM);
            if (bMEM != nullptr)
                cudaFree(bMEM);
            if (cMEM != nullptr)
                delete[] (float*)cMEM;
            if (gMEM != nullptr)
                cudaFree(gMEM);
            cudaDeviceSynchronize();
#ifdef __DEBUG__
            printf("%s delete\n", name.c_str());
#endif
        }
        void run();
        size_t setParam(float* buffer);
    };


    class dense : public layer
    {
    protected:
        float* wMEM = nullptr;
        float* bMEM = nullptr;
        bool useBias;
        int inNodeSize;
        size_t weightSize;
        size_t biasSize;

    public:
        dense() = delete;
        dense(const dense&) = delete;
        dense(layer* p, int numNode, bool use_bias = true, dim3 blockSize = dim3(1, 1, 0));
        ~dense() {
            if (wMEM != nullptr)
                cudaFree(wMEM);
            if (bMEM != nullptr)
                cudaFree(bMEM);
            if (cMEM != nullptr)
                delete[] (float*)cMEM;
            if (gMEM != nullptr)
                cudaFree(gMEM);
            cudaDeviceSynchronize();
#ifdef __DEBUG__
            printf("%s delete\n", name.c_str());
#endif
        }
        void run();
        size_t setParam(float* buffer);
    };


    class maxpool2d : public layer
    {
    protected:
        float* wMEM = nullptr;
        float* bMEM = nullptr;
        int padSize;
        int stride;
        int kSize;

    public:
        maxpool2d() = delete;
        maxpool2d(const maxpool2d&) = delete;
        maxpool2d(layer* p, int ks, int stride = 1, int pad = 0, dim3 blockSize = dim3(1, 1, 1));
        ~maxpool2d() {
            if (wMEM != nullptr)
                cudaFree(wMEM);
            if (bMEM != nullptr)
                cudaFree(bMEM);
            if (cMEM != nullptr)
                delete[] (float*)cMEM;
            if (gMEM != nullptr)
                cudaFree(gMEM);
            cudaDeviceSynchronize();
#ifdef __DEBUG__
            printf("%s delete\n", name.c_str());
#endif
        }
        void run();
        size_t setParam(float* buffer);
    };


    class relu : public layer
    {
    protected:

    public:
        relu() = delete;
        relu(const relu&) = delete;
        relu(layer* p, bool inplace_ = true, dim3 blockSize = dim3(1, 0, 0));
        void run();
        size_t setParam(float* buffer);
    };


    class relu6 : public layer
    {
    protected:

    public:
        relu6() = delete;
        relu6(const relu6&) = delete;
        relu6(layer* p, bool inplace_ = true, dim3 blockSize = dim3(1, 0, 0));
        void run();
        size_t setParam(float* buffer);
    };

    
    class leakyrelu : public layer
    {
    protected:
        float negativeSlope = 0;

    public:
        leakyrelu() = delete;
        leakyrelu(const leakyrelu&) = delete;
        leakyrelu(layer *p, float negativeSlope, bool inplace_ = true, dim3 blockSize = dim3(1, 0, 0));
        void run();
        size_t setParam(float* buffer);
    };


    class softmax : public layer
    {
    protected:

    public:
        softmax() = delete;
        softmax(const softmax&) = delete;
        softmax(layer* p, bool inplace_ = true, dim3 blockSize = dim3(1, 1, 0));
        void run();
        size_t setParam(float* buffer);
    };


    class batchnorm : public layer
    {
    protected:
        float* pMEM = nullptr;
        size_t paramSize = 0;
        float epsilon = EPSILON;

    public:
        batchnorm() = delete;
        batchnorm(const batchnorm&) = delete;
        batchnorm(layer* p, float epsilon_ = EPSILON, bool inplace_ = true, dim3 blockSize = dim3(1, 1, 1));
        ~batchnorm() {
            if (pMEM != nullptr)
                cudaFree(pMEM);
            if (cMEM != nullptr)
                delete[] (float*)cMEM;
            if (gMEM != nullptr)
                cudaFree(gMEM);
            cudaDeviceSynchronize();
#ifdef __DEBUG__
            printf("%s delete\n", name.c_str());
#endif
        }
        void run();
        size_t setParam(float* buffer);
    };


    class add : public layer
    {
    protected:
        layer* srcLayer;

    public:
        add() = delete;
        add(const add&) = delete;
        add(layer* p, layer* src, bool inplace_ = true, dim3 blockSize = dim3(1, 0, 0));
        void run();
        size_t setParam(float* buffer);
    };


    // Calculate average for each channel
    class globalavg : public layer
    {
    protected:

    public:
        globalavg() = delete;
        globalavg(const globalavg&) = delete;
        globalavg(layer* p, dim3 blockSize = dim3(1, 1, 0));
        void run();
        size_t setParam(float* buffer);
    };

    /**
     * @brief Sequential module to build model [ sequential.cu ]
     * 
     */
    class sequential
    {
    protected:
        std::vector<layer*> seq;
        size_t total_gpu_memory_size = 0;

    public:
        sequential() {}
        ~sequential() {
            for (int i = 0; i < seq.size(); ++i) {
                if(i == seq.size() - 1)
                    break;
#ifdef __DEBUG__
                printf("Start delete\t %s(%d)\t layer\n", seq[i]->getName().c_str(), i);
#endif
                delete seq[i];
            }
        }
        void add(layer *l);
        void summary();
        size_t getSize() { return seq.size(); }
        size_t getMemSize() { return total_gpu_memory_size * sizeof(float); }
        layer* operator[](int i);
        layer* back() { return seq.back(); } ;
        bool loadWeight(const std::string &filePath);
        void inference(float* x, float* y);
        void getLayerfeature(float* x, float* y, const size_t &layer_idx);
        void getArgumentReference();
    };
}

#endif