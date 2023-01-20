#pragma once

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include <torch/script.h>
#include <torch/torch.h>

#include "config.cuh"
#include "math.cuh"

#define TRY(err) {                                                     \
	if (err != cudaSuccess) {                                          \
		printf("[CUDA] operation failed line %d\n", __LINE__);         \
		printf("[CUDA] %s\n", cudaGetErrorString(cudaGetLastError())); \
		exit(EXIT_FAILURE);                                            \
	}                                                                  \
}

void gen_poses(
	const float theta_a, const float theta_b, const float phi, const float radius,
	Mat4 *poses, const int n
);

__global__ void gen_rays(ray_t *rays, mat4_t *pose, config_t* config);
__global__ void gen_samples(ray_t *rays, float *rays_pos, float *rays_dir, config_t* config);

__global__ void accumulate_weights(float* sigma, float* weights, config_t* config);
__global__ void accumulate_color(float* weights, float* rgb, float* pix, config_t* config);

void init_rays(
	ray_t *d_rays, mat4_t *d_pose, torch::Tensor t_pos, torch::Tensor t_dir,
	config_t* config
);
void nerf_inference(
	torch::jit::Module t_phi_p, torch::jit::Module t_phi_d, torch::jit::Module t_nerf,
	torch::Tensor t_pos, torch::Tensor t_dir, torch::Tensor t_sig, torch::Tensor t_rgb,
	config_t* config
);
void render_volume(
	torch::Tensor t_sig, torch::Tensor t_rgb, torch::Tensor t_wei, torch::Tensor t_pix,
	config_t* config
);