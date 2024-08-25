ai_framework: main.o basic.o layer_basic.o sequential.o layer_input.o layer_conv2d.o layer_dwconv2d.o layer_pwconv2d.o layer_dense.o layer_maxpool2d.o layer_relu.o layer_relu6.o layer_leakyrelu.o layer_softmax.o layer_batchnorm.o layer_add.o layer_globalavg.o
	@nvcc -o ai_framework *.o -g -m64 -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 -lcublas -lcudnn --compiler-options -Wall `pkg-config opencv4 --cflags --libs` -Xcompiler -fopenmp

basic.o:
	@nvcc -c -g basic.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_basic.o:
	@nvcc -c -g layer_basic.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

sequential.o:
	@nvcc -c -g sequential.cpp -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_input.o:
	@nvcc -c -g layer_input.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_conv2d.o:
	@nvcc -c -g layer_conv2d.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_dwconv2d.o:
	@nvcc -c -g layer_dwconv2d.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_pwconv2d.o:
	@nvcc -c -g layer_pwconv2d.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_dense.o:
	@nvcc -c -g layer_dense.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_maxpool2d.o:
	@nvcc -c -g layer_maxpool2d.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_relu.o:
	@nvcc -c -g layer_relu.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_relu6.o:
	@nvcc -c -g layer_relu6.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_leakyrelu.o:
	@nvcc -c -g layer_leakyrelu.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_softmax.o:
	@nvcc -c -g layer_softmax.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_batchnorm.o:
	@nvcc -c -g layer_batchnorm.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_add.o:
	@nvcc -c -g layer_add.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

layer_globalavg.o:
	@nvcc -c -g layer_globalavg.cu -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`

main.o:
	@nvcc -c -g main.cpp -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 `pkg-config opencv4 --cflags --libs`
