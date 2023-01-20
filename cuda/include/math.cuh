#pragma once

#include <cmath>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define PI 3.141593f

typedef struct Vec3 {
    float x, y, z;

    __host__ __device__ __forceinline__ Vec3() { x = 0.; y = 0.; z = 0.; }
    __host__ __device__ __forceinline__ Vec3(const float _x, const float _y, const float _z) { x = _x; y = _y; z = _z; }

    __host__ __device__ __forceinline__ Vec3 operator+(const float s) const { return Vec3(x + s, y + s, z + s); }
    __host__ __device__ __forceinline__ Vec3 operator-(const float s) const { return Vec3(x - s, y - s, z - s); }
    __host__ __device__ __forceinline__ Vec3 operator*(const float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ __forceinline__ Vec3 operator/(const float s) const { return Vec3(x / s, y / s, z / s); }

    __host__ __device__ __forceinline__ Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ __forceinline__ Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ __forceinline__ Vec3 operator*(const Vec3 &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ __forceinline__ Vec3 operator/(const Vec3 &v) const { return Vec3(x / v.x, y / v.y, z / v.z); }

    __host__ __device__ __forceinline__ void operator+=(const float s) { x += s; y += s; z += s; }
    __host__ __device__ __forceinline__ void operator-=(const float s) { x -= s; y -= s; z -= s; }
    __host__ __device__ __forceinline__ void operator*=(const float s) { x *= s; y *= s; z *= s; }
    __host__ __device__ __forceinline__ void operator/=(const float s) { x /= s; y /= s; z /= s; }

    __host__ __device__ __forceinline__ void operator+=(const Vec3 &v) { x += v.x; y += v.y; z += v.z; }
    __host__ __device__ __forceinline__ void operator-=(const Vec3 &v) { x -= v.x; y -= v.y; z -= v.z; }
    __host__ __device__ __forceinline__ void operator*=(const Vec3 &v) { x *= v.x; y *= v.y; z *= v.z; }
    __host__ __device__ __forceinline__ void operator/=(const Vec3 &v) { x /= v.x; y /= v.y; z /= v.z; }

	__host__ __device__ __forceinline__ float magnitude() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__ __forceinline__ Vec3  normalize() const { return *this / this->magnitude(); }
} vec3_t;

typedef struct Mat4 {
    float data[4][4];
    
    __host__ __device__ __forceinline__ Mat4() {
        data[0][0] = 0.f; data[0][1] = 0.f; data[0][2] = 0.f; data[0][3] = 0.f;
        data[1][0] = 0.f; data[1][1] = 0.f; data[1][2] = 0.f; data[1][3] = 0.f;
        data[2][0] = 0.f; data[2][1] = 0.f; data[2][2] = 0.f; data[2][3] = 0.f;
        data[3][0] = 0.f; data[3][1] = 0.f; data[3][2] = 0.f; data[3][3] = 0.f;
    }
    __host__ __device__ __forceinline__ Mat4(
        const float a, const float b, const float c, const float tx,
        const float d, const float e, const float f, const float ty,
        const float g, const float h, const float i, const float tz,
        const float j, const float k, const float l, const float tw
    ) {
        data[0][0] = a; data[0][1] = b; data[0][2] = c; data[0][3] = tx;
        data[1][0] = d; data[1][1] = e; data[1][2] = f; data[1][3] = ty;
        data[2][0] = g; data[2][1] = h; data[2][2] = i; data[2][3] = tz;
        data[3][0] = j; data[3][1] = k; data[3][2] = l; data[3][3] = tw;
    }

    __host__ __device__ __forceinline__ static Mat4 eye() {
        Mat4 ret = Mat4();
        ret.data[0][0] = 1.; ret.data[1][1] = 1.;
        ret.data[2][2] = 1.; ret.data[3][3] = 1.;
        return ret;
    }
    
    __host__ __device__ __forceinline__ static Mat4 turn(const float theta, const float phi, const float radius) {
        float ct = cosf(theta), st = sinf(theta);
        float cp = cosf(phi  ), sp = sinf(phi  );
        
        Mat4 rt = Mat4::eye();
        rt.data[0][0] = ct; rt.data[0][2] = -st;
        rt.data[2][0] = st; rt.data[2][2] =  ct;

        Mat4 rp = Mat4::eye();
        rp.data[1][1] = cp; rp.data[1][2] = -sp;
        rp.data[2][1] = sp; rp.data[2][2] =  cp;
        
        Mat4 tr = Mat4::eye();
        tr.data[1][3] = 0.02f * radius;
        tr.data[2][3] = radius;

        Mat4 fl = Mat4::eye();
        fl.data[0][0] = -1.f;
        fl.data[1][1] =  0.f; fl.data[1][2] = 1.f;
        fl.data[2][1] =  1.f; fl.data[2][2] = 0.f;

        return fl * rp * rt * tr;
    }

    __host__ __device__ __forceinline__ Mat4 T() const {
        Mat4 ret;
        ret.data[0][0] = data[0][0]; ret.data[0][1] = data[1][0]; ret.data[0][2] = data[2][0]; ret.data[0][3] = data[3][0];
        ret.data[1][0] = data[0][1]; ret.data[1][1] = data[1][1]; ret.data[1][2] = data[2][1]; ret.data[1][3] = data[3][1];
        ret.data[2][0] = data[0][2]; ret.data[2][1] = data[1][2]; ret.data[2][2] = data[2][2]; ret.data[2][3] = data[3][2];
        ret.data[3][0] = data[0][3]; ret.data[3][1] = data[1][3]; ret.data[3][2] = data[2][3]; ret.data[3][3] = data[3][3];
        return ret;
    }

	__host__ __device__ __forceinline__ Mat4 operator*(const Mat4 &M) const {
        Mat4 ret = Mat4();
        for (int k = 0; k < 4; ++k)
        for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            ret.data[i][j] += data[i][k] * M.data[k][j];
        return ret;
    }

    __host__ __device__ __forceinline__ static Vec3 dot(const Mat4 &M, const Vec3 &v) {
        auto a11 = M.data[0][0], a12 = M.data[1][0], a13 = M.data[2][0];
        auto a21 = M.data[0][1], a22 = M.data[1][1], a23 = M.data[2][1];
        auto a31 = M.data[0][2], a32 = M.data[1][2], a33 = M.data[2][2];
        
        return Vec3(
            v.x * a11 + v.y * a21 + v.z * a31,
            v.x * a12 + v.y * a22 + v.z * a32,
            v.x * a13 + v.y * a23 + v.z * a33
        );
    }
} mat4_t;

typedef struct Ray {
    vec3_t ori, dir;

    __host__ __device__ __forceinline__ Ray(const vec3_t &_ori, const vec3_t &_dir) { ori = _ori; dir = _dir; }
    __host__ __device__ __forceinline__ vec3_t evaluate(const float t) { return ori + dir * t; }
} ray_t;