#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#include <png++/png.hpp>
#include <png++/image.hpp>
#include <png++/rgb_pixel.hpp>

#include "argparse.cuh"
#include "config.cuh"
#include "math.cuh"
#include "nerf.cuh"

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Path to phi_p, phi_d, and nerf models are required.");
		return EXIT_FAILURE;
	}

	argparse::ArgumentParser parser("NeRF", "DistillNeRF Renderer");
	parser.add_argument("-p", "--phi_p",  "positional encoder for position",  true );
	parser.add_argument("-d", "--phi_d",  "positional encoder for direction", true );
	parser.add_argument("-n", "--nerf",   "nerf model",                       true );
	parser.add_argument("-f", "--frames", "number for frames to render",      false);
	parser.enable_help();

	auto err = parser.parse(argc, argv);
	if (err) {
		printf("%s\n", err);
		return EXIT_FAILURE;
	}

	format_t      FORMAT      = Format(256, 256);
	camera_t      CAMERA      = Camera(0.69f, 2.f, 6.f, FORMAT);
	sampling_t    SAMPLING    = Sampling(64, 64);
	performance_t PERFORMANCE = Performance(128, 256, 4096, FORMAT, SAMPLING);
	config_t      CONFIG      = { FORMAT, CAMERA, SAMPLING, PERFORMANCE };

	mat4_t *h_poses = (mat4_t*)malloc(sizeof(mat4_t) * parser.get<int>("frames"));
	gen_poses(0.f, 2.f * PI, 0.f, 4.f, h_poses, parser.get<int>("frames"));

	torch::InferenceMode guard;
	auto t_phi_p = torch::jit::load(parser.get<char*>("phi_p"), torch::kCUDA);
	auto t_phi_d = torch::jit::load(parser.get<char*>("phi_d"), torch::kCUDA);
	auto t_nerf  = torch::jit::load(parser.get<char*>("nerf" ), torch::kCUDA);

	auto t_opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	auto t_pos = torch::zeros({ CONFIG.format.pixels, CONFIG.sampling.coarse, 3 }, t_opt);
	auto t_dir = torch::zeros({ CONFIG.format.pixels, CONFIG.sampling.coarse, 3 }, t_opt);
	auto t_sig = torch::zeros({ CONFIG.format.pixels, CONFIG.sampling.coarse    }, t_opt);
	auto t_rgb = torch::zeros({ CONFIG.format.pixels, CONFIG.sampling.coarse, 3 }, t_opt);
	auto t_wei = torch::zeros({ CONFIG.format.pixels, CONFIG.sampling.coarse    }, t_opt);
	auto t_pix = torch::zeros({ CONFIG.format.pixels                        , 3 }, t_opt);

	mat4_t *d_pose;
	TRY(cudaMalloc((void**)&d_pose, sizeof(mat4_t)));
	
	ray_t *d_rays;
	TRY(cudaMalloc((void**)&d_rays, sizeof(ray_t) * CONFIG.format.pixels));
	
	float start, sec;
	for (size_t i = 0; i < parser.get<int>("frames"); ++i) {
		start = clock();
		{
			TRY(cudaMemcpy(d_pose, &h_poses[i], sizeof(mat4_t), cudaMemcpyHostToDevice));
			
			init_rays(d_rays, d_pose, t_pos, t_dir, &CONFIG);
			nerf_inference(t_phi_p, t_phi_d, t_nerf, t_pos, t_dir, t_sig, t_rgb, &CONFIG);
			render_volume(t_sig, t_rgb, t_wei, t_pix, &CONFIG);
		}
		sec = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
		printf("FPS: %.2lf\r", 1. / sec);
		
		float* buffer = (t_pix.clip(0.f, 1.f) * 255.f).clone().to(torch::kCPU).data_ptr<float>();

		png::image<png::rgb_pixel> image(CONFIG.format.width, CONFIG.format.height);
		for (int y = 0; y < CONFIG.format.height; ++y)
		for (int x = 0; x < CONFIG.format.width;  ++x)
			image.set_pixel(x, y, png::rgb_pixel(
				(unsigned char)buffer[(x + y * CONFIG.format.width) * 3 + 0],
				(unsigned char)buffer[(x + y * CONFIG.format.width) * 3 + 1],
				(unsigned char)buffer[(x + y * CONFIG.format.width) * 3 + 2]
			));

		std::string path = "frames/nerf_" + std::to_string(i) + std::string(".png");
		image.write(path);
	}
	printf("FPS: %.2lf\n", 1. / sec);

	TRY(cudaFree(d_pose));
	TRY(cudaFree(d_rays));
	TRY(cudaDeviceReset());

	free(h_poses);

	return EXIT_SUCCESS;
}