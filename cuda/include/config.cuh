#pragma once

#include <cuda.h>

typedef struct Format {
    int width, height, pixels;

    Format(const int _width, const int _height) {
        width = _width;
        height = _height;
        pixels = _width * _height;
    }
} format_t;

typedef struct Camera {
    int focal, near, far;

    Camera(const float _fov, const float _near, const float _far, const format_t format) {
        focal = .5f * (float)format.width / tanf(.5f * _fov);
        near = _near;
        far = _far;
    }
} camera_t;

typedef struct Sampling {
    int coarse, fine, total;

    Sampling(const int _coarse, const int _fine) {
        coarse = _coarse;
        fine = _fine;
        total = _coarse + _fine;
    }
} sampling_t;

typedef struct Performance {
    dim3 r_blocks, r_threads_per_block;
    dim3 s_blocks, s_threads_per_block;
    int batch_size;

    Performance(const int r, const int s, const int _batch_size, const format_t format, const sampling_t sampling) {
        r_blocks = { r, r, 1 };
        r_threads_per_block = {
            format.width  / r + (int)(format.width  % r > 0),
            format.height / r + (int)(format.height % r > 0),
            1
        };

        s_blocks = { s, s, 1 };
        s_threads_per_block = {
            format.width  / s + (int)(format.width  % s > 0),
            format.height / s + (int)(format.height % s > 0),
            sampling.coarse
        };

        batch_size = _batch_size;
    }
} performance_t;

typedef struct Config { format_t format; camera_t camera; sampling_t sampling; performance_t performance; } config_t;