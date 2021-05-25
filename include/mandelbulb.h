#ifndef MANDELBULB_H
#define MANDELBULB_H

#include <stdio.h>

#include "global.h"
#include "vec3d.h"
#include "camera.h"

class Mandelbulb {
public:
    Mandelbulb();
    ~Mandelbulb();

    __host__ __device__ float4 getPixel(int x, int y, Camera* camera);

    __host__ __device__ float getDist(const Vec3d& v);
    __host__ __device__ Vec3d getNormal(const Vec3d& v, float d);
};

#endif