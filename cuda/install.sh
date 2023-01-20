#!/usr/bin/sh
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive --jobs 0

sudo USE_CUDA=1 CFLAGS="-Ofast -std=c++14" CC=cc USE_CUDNN=1 USE_FBGEMM=1 USE_KINETO=1 USE_NUMPY=1 BUILD_TEST=0 USE_MKLDNN=1 USE_NNPACK=1 USE_QNNPACK=1 USE_DISTRIBUTED=1 USE_TENSORPIPE=1 USE_GLOO=1 USE_MPI=1 USE_SYSTEM_NCCL=0 BUILD_CAFFE2_OPS=1 BUILD_CAFFE2=1 BUILD_TORCH=1 BUILD_SHARED_LIBS=1 USE_IBVERBS=1 USE_OPENCV=1 USE_OPENMP=1 USE_FFMPEG=1 USE_LEVELDB=1 USE_LMDB=1 BUILD_BINARY=1 ATEN_AVX512_256=TRUE TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=/usr/local/cuda CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc CUDACXX=/usr/local/cuda/bin/nvcc LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib python3 setup.py install

cd ..
sudo rm -rf pytorch

python3 -c "import torch; print(f'[CUDA] Available: {torch.cuda.is_available()}'); torch.zeros((1, 1), device='cuda:0')"
