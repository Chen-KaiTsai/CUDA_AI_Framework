# CUDA_AI_Framework
A CUDA C implementation of a AI Framework. Only support inference with Nvidia GPUs (Desktop/WorkStation/Jetson).

# Document
Please refer to the following HackMD post for the detail of the framework.
> https://hackmd.io/@Erebustsai/Sku_EMMr2
*Notice that this document is for OpenCL version of this framework; however, the interface is the same.*

# Support \/ Limitation

## Supported Layers

* 2d Convolution Layer
* 2d Point-wise Convolution Layer
* 2d Depth-wise Convolution Layer
* 2d Max Pooling Layer
* Bach Normalization Layer
* Fully Connective Layer
* Concatenation Operation
* Global Average Operation
* ReLU
* ReLU6
* LeakyReLU
* Softmax

## Helper Functions

* Read img as model input
* Device info report
* `IoU` \(for YOLO\)
* `NMS` \(for YOLO\)
* `getArgumentReference` \(This function can be used to list candidate value for thread block dimensions\)
* `getLayerfeature` \(This can be used to check if the model work as expected on a certain layer\)

# How to use
1. Make sure you have CUDA and cuDNN installed.
2. Change the `Makefile` for CUDA sample common headers. Mine is in `/usr/local/cuda/samples/Common/`. Please download CUDA samples on Nvidia's website.
3. Change the `Makefile` for your GPU compute capabilities.
4. Export weight from **pytorch** with the following code snippet.
```python
if EXPORT_WEIGHT:
    model.eval()
    if not LOAD_MODEL:
        print("----------Warning!----------\n Not Loading ANY MODEL!\nThe output binary file might not be meaningful.")
    
    print("Exporting weight file " + LOAD_PATH)
    LOAD_PATH = LOAD_PATH.replace('pt', 'bin')
    with open(LOAD_PATH, "wb") as file:
        for param_name in model.state_dict():
            if param_name.find("num_batches_tracked") != -1:
                continue
            layer_weight = model.state_dict()[param_name].flatten().numpy()
            for weight in layer_weight:
                file.write(weight)
    print(LOAD_PATH + " Weight file exported")
    exit()
```
5. `make` and run the output binary.
