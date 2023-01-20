PYTHON_VERSION=3.8

NVCC=/usr/local/cuda/bin/nvcc
PROFILER=/usr/local/cuda/bin/nvprof

LIBTORCH=/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch

ROOT=cuda
INSTALL=${ROOT}/install.sh

INC=-I ${ROOT}/include
SRC=${ROOT}/src/nerf.cu ${ROOT}/src/main.cu
TARGET=${ROOT}/NeRF

LIBTORCH_INCPATH+=-I${LIBTORCH}/include
LIBTORCH_INCPATH+=-I${LIBTORCH}/include/torch/csrc/api/include
LIBTORCH_INCPATH+=-I${LIBTORCH}/include/TH
LIBTORCH_INCPATH+=-I${LIBTORCH}/include/THC

CUDA_LIBS+=${LIBTORCH}/lib/libc10_cuda.so
CUDA_LIBS+=${LIBTORCH}/lib/libc10.so
CUDA_LIBS+=${LIBTORCH}/lib/libtorch_cpu.so
CUDA_LIBS+=${LIBTORCH}/lib/libtorch_cuda.so
CUDA_LIBS+=${LIBTORCH}/lib/libtorch.so

LIBS+=${CUDA_LIBS} `libpng-config --cflags --ldflags`
FLAGS=-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=1 -arch=sm_86 -use_fast_math -O3

PRELOAD_CUDA_LIBS=${subst ${subst ,, },:,${CUDA_LIBS}}

BASE=res/NanoNeRF/lego/jit
PHI_P=${BASE}/phi_p.pt
PHI_D=${BASE}/phi_d.pt
NERF=${BASE}/nerf.pt

all: build run

install:
	./${INSTALL}

build:
	${NVCC} ${INC} ${SRC} -o ${TARGET} ${LIBTORCH_INCPATH} ${LIBS} ${FLAGS}

run:
	mkdir -p frames
	LD_PRELOAD=${PRELOAD_CUDA_LIBS} ./${TARGET} --phi_p ${PHI_P} --phi_d ${PHI_D} --nerf ${NERF} --frames 80
	ffmpeg -y -framerate 10 -i frames/nerf_%d.png -loop 0 res/frames.gif
	rm -rf frames

ldd:
	LD_PRELOAD=${PRELOAD_CUDA_LIBS} ldd ${TARGET}

profile:
	${PROFILER} ./${TARGET}