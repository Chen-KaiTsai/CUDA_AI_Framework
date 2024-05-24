#include "framework.cuh"

#include <future>
#include <thread>
#include <atomic>

#include <chrono>

std::atomic<bool> stopFlag;

void bufferLoader(float* data) {
    // OpenCV HWC -> CHW
    for(int b = 0; b < 1; ++b) {
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < 224; ++y) {
                for (int x = 0; x < 224; ++x)
                    data[b * 3 * 224 * 224 + c * 224 * 224 + y * 224 + x] = 1.0f;
            }
        }
    }
}

void bufferedImageLoader(cv::Mat& image, float* data)
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
                data[c * image.rows * image.cols + y * image.cols + x] = (float)If32.at<cv::Vec3f>(y, x)[c] / 255.0f;
            }
        }
    }
}

void hostStopper() {
    getchar();
    stopFlag = true;
}

int main()
{
    framework::getDeviceName();
    framework::getDeviceInfo(0);

    std::unique_ptr<float[]> predict = std::make_unique<float[]>(1 * 2);
    std::unique_ptr<float[]> data = std::make_unique<float[]>(1 * 3 * 224 * 224);

    std::future<void> backgroundThread = std::async(std::launch::async, bufferLoader, data.get());

    framework::sequential model;

    model.add(new framework::input({1, 3, 224, 224}));
    
    model.add(new framework::conv2d(model.back(), 32, 3, 2, 1, false, 1, {16, 16, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {16, 16, 4}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    // 1
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {16, 16, 4})); //4
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {16, 16, 4}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::pwconv2d(model.back(), 16, false, 1, {16, 16, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {16, 16, 4}));

    // 2
    model.add(new framework::pwconv2d(model.back(), 96, false, 1, {16, 16, 4})); //9
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {16, 16, 4}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::dwconv2d(model.back(), 3, 2, 1, false, 1, {8, 8, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {8, 8, 16}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::pwconv2d(model.back(), 24, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    // 3
    model.add(new framework::pwconv2d(model.back(), 144, false, 1, {14, 14, 4})); //17
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::pwconv2d(model.back(), 24, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::add(model.back(), model[16], true, {896})); 

    // 4
    model.add(new framework::pwconv2d(model.back(), 144, false, 1, {8, 8, 16})); //26
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {8, 8, 16}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::dwconv2d(model.back(), 3, 2, 1, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {1008}));
    model.add(new framework::pwconv2d(model.back(), 32, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));

    // 6
    model.add(new framework::pwconv2d(model.back(), 192, false, 1, {4, 4, 64})); //34
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {4, 4, 64}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {4, 4, 64}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {4, 4, 64}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::pwconv2d(model.back(), 32, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::add(model.back(), model[33], true, {896}));

    // 7
    model.add(new framework::pwconv2d(model.back(), 192, false, 1, {4, 4, 64})); //43
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {4, 4, 64}));
    model.add(new framework::relu6(model.back(), true, {1024}));
    model.add(new framework::dwconv2d(model.back(), 3, 2, 1, false, 1, {7, 7, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::relu6(model.back(), true, {896}));
    model.add(new framework::pwconv2d(model.back(), 64, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    // 8
    model.add(new framework::pwconv2d(model.back(), 384, false, 1, {14, 14, 4})); //51
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {896}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {896}));
    model.add(new framework::pwconv2d(model.back(), 64, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::add(model.back(), model[50], true, {896}));

    // 10
    model.add(new framework::pwconv2d(model.back(), 384, false, 1, {14, 14, 4})); //60
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {896}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {896}));
    model.add(new framework::pwconv2d(model.back(), 64, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::add(model.back(), model[59], true, {896}));

    // 11
    model.add(new framework::pwconv2d(model.back(), 384, false, 1, {14, 14, 4})); //69
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {896}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {896}));
    model.add(new framework::pwconv2d(model.back(), 96, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));

    // 13
    model.add(new framework::pwconv2d(model.back(), 576, false, 1, {14, 14, 4})); //77
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {1008}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {1008}));
    model.add(new framework::pwconv2d(model.back(), 96, false, 1, {14, 14, 4}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::add(model.back(), model[76], true, {896}));

    // 14
    model.add(new framework::pwconv2d(model.back(), 576, false, 1, {14, 14, 4})); //86
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {14, 14, 4}));
    model.add(new framework::relu6(model.back(), true, {1008}));
    model.add(new framework::dwconv2d(model.back(), 3, 2, 1, false, 1, {7, 7, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::relu6(model.back(), true, {1008}));
    model.add(new framework::pwconv2d(model.back(), 160, false, 1, {7, 7, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));

    // 16
    model.add(new framework::pwconv2d(model.back(), 960, false, 1, {7, 7, 16})); //94
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::relu6(model.back(), true, {980}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {7, 7, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::relu6(model.back(), true, {980}));
    model.add(new framework::pwconv2d(model.back(), 160, false, 0, {7, 7, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::add(model.back(), model[93], true, {980}));

    // 17
    model.add(new framework::pwconv2d(model.back(), 960, false, 0, {7, 7, 16})); //103
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::relu6(model.back(), true, {980}));
    model.add(new framework::dwconv2d(model.back(), 3, 1, 1, false, 1, {7, 7, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::relu6(model.back(), true, {980}));
    model.add(new framework::pwconv2d(model.back(), 320, false, 0, {7, 7, 16}));
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));

    // 18
    model.add(new framework::pwconv2d(model.back(), 1280, false, 0, {7, 7, 16})); //111
    model.add(new framework::batchnorm(model.back(), 1e-5, true, {7, 7, 16}));
    model.add(new framework::relu6(model.back(), true, {980}));

    model.add(new framework::globalavg(model.back(), {640, 1, 1})); //114

    model.add(new framework::dense(model.back(), 2, true, {2, 1, 1})); //115

    model.summary();

    model.getArgumentReference();
    
    model.loadWeight("modified_no_dropout_mobilenetv2_weight_20_epochs.bin");

    printf("total gpu memory size : %ld bytes(not include weight memory size)\n", model.getMemSize());

    backgroundThread.get();

    stopFlag = false;
    std::thread stopper(hostStopper);

    int epoch = 0;
    while(true) {
        if(stopFlag)
            break;
        printf("\n\n------------------------------ Epoch %d Start------------------------------\n\n", epoch++);
        auto start = std::chrono::steady_clock::now();
        model.inference(data.get(), predict.get());
        auto end = std::chrono::steady_clock::now();
        for (int i = 0; i < 2; ++i)
            printf("[%d] : %+03.4f\n", i, predict[i]);
        printf("\n\n------------------------------ Epoch END ------------------------------\n\n");
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Inference Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
        
    }

    stopper.join();

    return 0;
}

