#include "nerf.cuh"

void gen_poses(const float theta_a, const float theta_b, const float phi, const float radius, Mat4 *poses, const int n) {
	for (int i = 0; i < n; ++i) {
		float theta = theta_a * (1.f - (float)i / (float)n) + theta_b * ((float)i / (float)n);
		poses[i] = Mat4::turn(theta, phi, radius);
	}
}

__global__ void gen_rays(ray_t *rays, mat4_t *pose, config_t *config) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (id_x >= config->format.width || id_y >= config->format.height) return;

	int id = id_x + id_y * config->format.width;

	float rx = ((float)id_x - .5f * (float)config->format.width ) / config->camera.focal;
	float ry = ((float)id_y - .5f * (float)config->format.height) / config->camera.focal;
	float rz = 1.f;

	rays[id].ori = Vec3(pose->data[0][3], pose->data[1][3], pose->data[2][3]);
	rays[id].dir = Mat4::dot(*pose, Vec3(rx, -ry, -rz)).normalize();
}

__global__ void gen_samples(ray_t *rays, float *rays_pos, float *rays_dir, config_t *config) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (id_x >= config->format.width || id_y >= config->format.height) return;

	int rid = id_x + id_y * config->format.width;
	int sid = rid * config->sampling.coarse * 3 + threadIdx.z * 3;

	float t = config->camera.near + (float)threadIdx.z * (config->camera.far - config->camera.near) / (float)config->sampling.coarse;
	vec3_t rp = rays[rid].evaluate(t);

	rays_pos[sid + 0] = rp.x;
	rays_pos[sid + 1] = rp.y;
	rays_pos[sid + 2] = rp.z;

	rays_dir[sid + 0] = rays[rid].dir.x;
	rays_dir[sid + 1] = rays[rid].dir.y;
	rays_dir[sid + 2] = rays[rid].dir.z;
}

__global__ void accumulate_weights(float* sigma, float* weights, config_t *config) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (id_x >= config->format.width || id_y >= config->format.height) return;

	int rid = id_x + id_y * config->format.width;
	int sid = rid * config->sampling.coarse + threadIdx.z;

	float ta = config->camera.near + (float)(threadIdx.z + 0) * (config->camera.far - config->camera.near) / (float)config->sampling.coarse;
	float tb = config->camera.near + (float)(threadIdx.z + 1) * (config->camera.far - config->camera.near) / (float)config->sampling.coarse;

	float delta = tb - ta;
	float alpha = 1.f - __expf(-sigma[sid] * delta);
	
	weights[sid] = 1.f - alpha + 1e-10;
	__syncthreads();

	if (threadIdx.z == 0) {
		int sample = 0;
		
		float tmp;
		float cumprod = 1.f;
		while (sample < config->sampling.coarse) {
			tmp = cumprod * weights[rid * config->sampling.coarse + sample];
			weights[rid * config->sampling.coarse + sample] = cumprod;

			cumprod = tmp;
			sample++;
		}
	}
	
	__syncthreads();
	weights[sid] *= alpha;
	if (weights[sid] == CUDART_NAN_F) weights[sid] = 0.;
	if (weights[sid] == CUDART_INF_F) weights[sid] = 1.;
}

__global__ void accumulate_color(float* weights, float* rgb, float* pix, config_t *config) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (id_x >= config->format.width || id_y >= config->format.height) return;

	int rid = id_x + id_y * config->format.width;
	int sid = rid * config->sampling.coarse;

	float r = 0.f;
	float g = 0.f;
	float b = 0.f;

	for (int sample = 0; sample < config->sampling.coarse; ++sample) {
		r +=  weights[sid + sample] * rgb[sid * 3 + sample * 3 + 0];
		g +=  weights[sid + sample] * rgb[sid * 3 + sample * 3 + 1];
		b +=  weights[sid + sample] * rgb[sid * 3 + sample * 3 + 2];
	}

	pix[rid * 3 + 0] = r;
	pix[rid * 3 + 1] = g;
	pix[rid * 3 + 2] = b;
}

void init_rays(ray_t *d_rays, mat4_t *d_pose, torch::Tensor t_pos, torch::Tensor t_dir, config_t *config) {
	gen_rays   <<<config->performance.r_blocks, config->performance.r_threads_per_block>>>(d_rays, d_pose, config);
	gen_samples<<<config->performance.s_blocks, config->performance.s_threads_per_block>>>(d_rays, t_pos.data_ptr<float>(), t_dir.data_ptr<float>(), config);
	TRY(cudaDeviceSynchronize());
}

void nerf_inference(
	torch::jit::Module t_phi_p, torch::jit::Module t_phi_d, torch::jit::Module t_nerf,
	torch::Tensor t_pos, torch::Tensor t_dir, torch::Tensor t_sig, torch::Tensor t_rgb,
	config_t *config
) {
	for (int start = 0; start < config->format.pixels; start = start + config->performance.batch_size) {
		torch::indexing::Slice slice(start, min(start + config->performance.batch_size, config->format.pixels - 1));
		int n = slice.stop() - slice.start();

		auto t_phi_p_out = t_phi_p.forward({ t_pos.index({ slice }).view({ n * config->sampling.coarse, -1 }) });
		auto t_phi_d_out = t_phi_d.forward({ t_dir.index({ slice }).view({ n * config->sampling.coarse, -1 }) });
		auto t_out = t_nerf.forward({ t_phi_p_out, t_phi_d_out }).toTuple()->elements();
		
		t_sig.index({ slice }) = t_out[0].toTensor().view({ n, config->sampling.coarse    });
		t_rgb.index({ slice }) = t_out[1].toTensor().view({ n, config->sampling.coarse, 3 });
	}
}

void render_volume(torch::Tensor t_sig, torch::Tensor t_rgb, torch::Tensor t_wei, torch::Tensor t_pix, config_t *config) {
	accumulate_weights<<<config->performance.r_blocks, config->performance.r_threads_per_block>>>(t_sig.data_ptr<float>(), t_wei.data_ptr<float>(), config);
	accumulate_color  <<<config->performance.s_blocks, config->performance.s_threads_per_block>>>(t_wei.data_ptr<float>(), t_rgb.data_ptr<float>(), t_pix.data_ptr<float>(), config);
	TRY(cudaDeviceSynchronize());
}