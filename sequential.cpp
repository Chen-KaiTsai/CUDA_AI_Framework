#include "framework.cuh"

#include <iostream>
#include <string>
#include <inttypes.h>

void floatToBinary(float f, std::string& str)
{
    union { float f; uint32_t i; } u;
    u.f = f;
    str.clear();

    for (int i = 0; i < 32; i++)
    {
        if (u.i % 2)  str.push_back('1');
        else str.push_back('0');
        u.i >>= 1;
    }

    // Reverse the string since now it's backwards
    std::string temp(str.rbegin(), str.rend());
    str = temp;
}

void listFactors(const dim3 gridSize)
{
    std::string xfactor;
    std::string yfactor;
    std::string zfactor;

    for (int i = 2; i <= gridSize.x; ++i) {
        if (gridSize.x % i == 0) {
            xfactor.append(std::to_string(i));
            xfactor.append(" ");
        }
    }

    for (int i = 2; i <= gridSize.y; ++i) {
        if (gridSize.y % i == 0) {
            yfactor.append(std::to_string(i));
            yfactor.append(" ");
        }
    }

    for (int i = 2; i <= gridSize.z; ++i) {
        if (gridSize.z % i == 0) {
            zfactor.append(std::to_string(i));
            zfactor.append(" ");
        }
    }

    printf("\n[\n%s,\n%s,\n%s\n]\n", xfactor.c_str(), yfactor.c_str(), zfactor.c_str());
}

/**
 * @brief Append layer to a sequential model.
 * 
 * @param l Layer address to append.
 */
void framework::sequential::add(layer *l)
{
    total_gpu_memory_size += (l->getGPUMemSize());
#ifdef __DEBUG__
    printf("Index : %ld\n\n", seq.size());
#endif
    seq.push_back(l);
}

framework::layer* framework::sequential::operator[](int i) { return seq[i]; }

/**
 * @brief List all layers in model
 * 
 */
void framework::sequential::summary() {
    size_t total_parameter_size = 0;
    printf("\n\nModel Summary\n\n");
    printf("#\tLayer\t\tOuput Shape[N,C,H,W]\t\tParameters\n");
    for(int i = 0; i < seq.size(); ++i)
    {
        printf("--------------------------------------------------------------------\n");
        shape_t s = seq[i]->getShape();
        std::string n = seq[i]->getName();
        size_t nParam = seq[i]->getNParam();
        total_parameter_size+=nParam;
        printf("[%d]\t%-10s\t[%3d, %3d, %3d, %3d]\t\t%ld\n", i, n.c_str(), s.N, s.C, s.H, s.W, nParam);
    }
    printf("--------------------------------------------------------------------\n\n");
    printf("                                                        %ld\n\n", total_parameter_size);
}

/**
 * @brief Load model weight with boundary checking
 * 
 * @param filePath Path to the model weight file
 * @return mdoel load weight status
 */
bool framework::sequential::loadWeight(const std::string &filePath) {
    FILE *pFile = fopen(filePath.c_str(), "rb");
    size_t nParam;
    if (pFile == nullptr) {
        printf("%s not found.\n", filePath.c_str());
        return false;
    }
    fseek(pFile, 0, SEEK_END);
    nParam = ftell(pFile) / sizeof(float);
    rewind(pFile);
    std::unique_ptr<float[]> buffer = std::make_unique<float[]>(nParam);
    int nRead = fread(buffer.get(), sizeof(float), nParam, pFile);
    fclose(pFile);
    if (nRead != nParam) {
        printf("%s : Number of byte is not correct.\n", filePath.c_str());
        return false;
    }

    int currLoad = 0, loaded;
    for (int i = 0; i < seq.size(); ++i) 
    {
#ifdef __DEBUG__
        printf("Load Layer Weight : %d\n", i);
#endif
        loaded = seq[i]->setParam(buffer.get() + currLoad);
        currLoad += loaded;
        if (currLoad > nParam) {
            printf("Weight not correct.\n");
            return false;
        }
        // seq[i]->setKernel();
    }
    printf("Model deployed.\n");
    return true;
}

/**
 * @brief Sequentially inference model layers
 * 
 * @param x model input
 * @param y model output
 */
void framework::sequential::inference(float* x, float* y)
{
    // Set & initial input buffer
    seq[0]->setCPUMem(x);
    seq.back()->setCPUMem(y);

    size_t size = seq.size();
    for (int i = 0; i < size; ++i) {
#ifdef __DEBUG__
        printf("Index Start : %s(%d)\n", seq[i]->getName().c_str(), i);
#endif
        seq[i]->run();
#ifdef __DEBUG__
        printf("Index End : %d\n\n", i);
#endif
    }
    seq.back()->detach();
}

/**
 * @brief [ Debug Only ] Print out top four outputs of each channel for a layer.
 * 
 * @param x model input
 * @param y model output
 * @param layer_idx idx of model layer to get output from
 */
void framework::sequential::getLayerfeature(float *x, float *y, const size_t &layer_idx)
{
    if(layer_idx < 0 || layer_idx > seq.size()) {
        printf("Layer index out of bound.\n");
    }

    seq[0]->setCPUMem(x);
    seq.back()->setCPUMem(y);

    for (int i = 0; i <= layer_idx; ++i) // Run till the debug layer
    {
        printf("Index Start : %d\n", i);
        seq[i]->run();
    }
    printf("\n\n--------------- DEBUG ---------------\n\n");
    framework::layer* debug_layer = seq[layer_idx];
    shape_t s = debug_layer->getShape();
    std::string n = debug_layer->getName();
    size_t nParam = debug_layer->getNParam();
    printf("%-10s\t[%3d, %3d, %3d, %3d]\t\t%ld\n", n.c_str(), s.N, s.C, s.H, s.W, nParam);
    
    debug_layer->detach();
    float* cMEM = debug_layer->getCPUMem();
    size_t outputSize = s.N * s.C * s.H * s.W;
    size_t batchSize = s.C * s.H * s.W;
    size_t fMapSize = s.H * s.W;
    printf("Layer output size: %ld\n", outputSize);
    printf("\n\nPrint out top four outputs of each channel for a layer\n\n");
    
    int batchIndex = 0;
    int channelIndex;
    for (size_t j = 0; j < outputSize; j += batchSize) 
    {
        printf("Batch\t\t%d\n", batchIndex++);
        channelIndex = 0;
        for (size_t l = 0; l < batchSize; l += fMapSize)
        {
            printf("Channel\t\t%d\n", channelIndex++);
            for (int k = 0; k < 4; ++k)
                printf("%+03.4f ", cMEM[j + k]);
            printf("...\n");
        }
        printf("\n\n\n");
    }
    printf("\n");
    return ;
}


void framework::sequential::getArgumentReference() 
{
    for(int i = 0; i < seq.size(); ++i)
    {
        printf("------------------------------------------------------------------------------\n");
        std::string n = seq[i]->getName();
        dim3 jobSize = seq[i]->getJobSize();
        printf("[%d]\t%-10s\tGrid : [%d, %d, %d]\t\t", i, n.c_str(), jobSize.x, jobSize.y, jobSize.z);
        listFactors(jobSize);
        printf("\n");
    }
    printf("------------------------------------------------------------------------------\n\n");
}
