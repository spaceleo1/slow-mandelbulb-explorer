#include "vec3d.h"

__host__ __device__ Vec3d::Vec3d(float x, float y, float z) :
    x(x), y(y), z(z) {}

__host__ __device__ Vec3d::~Vec3d() {}

__host__ __device__ Vec3d operator+(const Vec3d& a, const Vec3d& b) {
    return Vec3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ Vec3d operator-(const Vec3d& a, const Vec3d& b) {
    return Vec3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vec3d operator*(const Vec3d& a, float k) {
    return Vec3d(a.x * k, a.y * k, a.z * k);
}

__host__ __device__ float operator*(const Vec3d& a, const Vec3d& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float Vec3d::abs() {
    return sqrtf(x * x + y * y + z * z);
}

__host__ __device__ void Vec3d::normalize() {
    float norm = abs();
    x /= norm;
    y /= norm;
    z /= norm;
}