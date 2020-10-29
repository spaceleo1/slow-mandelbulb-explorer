#ifndef VEC3D_H
#define VEC3D_H

class Vec3d {
public:
    __host__ __device__ Vec3d(float x, float y, float z);
    __host__ __device__ ~Vec3d();

    friend __host__ __device__ Vec3d operator+(const Vec3d& a, const Vec3d& b);
    friend __host__ __device__ Vec3d operator-(const Vec3d& a, const Vec3d& b);
    friend __host__ __device__ Vec3d operator*(const Vec3d& a, float k);
    friend __host__ __device__ float operator*(const Vec3d& a, const Vec3d& b);

    __host__ __device__ float abs();
    __host__ __device__ void normalize();

    float x, y, z;
};

#endif