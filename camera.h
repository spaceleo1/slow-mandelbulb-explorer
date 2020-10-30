#ifndef CAMERA_H
#define CAMERA_H

#include "vec3d.h"

class Camera {
public:
    __host__ __device__ Camera(float x, float y, float z, float f, float angleX, float angleY);
    __host__ __device__ ~Camera();

    __host__ __device__ void rotateY(float angleShift);

    __host__ __device__ void rotateX(float angleShift);

    __host__ __device__ void moveForward(float shift);

    __host__ __device__ void moveSide(float shift);

    Vec3d pos;
    float f;
    float angleX;
    float angleY;
};

#endif