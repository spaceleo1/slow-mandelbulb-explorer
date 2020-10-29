#include "camera.h"

__host__ __device__ Camera::Camera(float x, float y, float z, float f, float angleX, float angleY) :
    pos(Vec3d(x, y, z)), f(f), angleX(angleX), angleY(angleY) {}

__host__ __device__ Camera::~Camera() {}