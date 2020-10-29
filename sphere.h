#ifndef SPHERE_H
#define SPHERE_H

#include <stdio.h>

#include "global.h"
#include "vec3d.h"
#include "camera.h"

class Sphere {
public:
    Sphere(float x, float y, float z, float r);
    ~Sphere();

    __host__ __device__ float4 getPixel(const Vec3d& c, float r, int x, int y, const Camera& camera);

    __host__ __device__ float getDist(const Vec3d& c, float r, const Vec3d& v);
    __host__ __device__ Vec3d getNormal(const Vec3d& c, float r, const Vec3d& v);

    Vec3d c;
    float r;
};

#endif