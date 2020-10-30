#include "camera.h"

__host__ __device__ Camera::Camera(float x, float y, float z, float f, float angleX, float angleY) :
    pos(Vec3d(x, y, z)), f(f), angleX(angleX), angleY(angleY) {}

__host__ __device__ Camera::~Camera() {}

__host__ __device__ void Camera::rotateY(float angleShift) {
    angleY += angleShift;
}

__host__ __device__ void Camera::rotateX(float angleShift) {
    angleX += angleShift;
}

__host__ __device__ void Camera::moveForward(float shift) {
    float shiftX = -sinf(angleY) * shift;
    float shiftZ = cosf(angleY) * shift;
    pos.x += shiftX;
    pos.z += shiftZ;
}

__host__ __device__ void Camera::moveSide(float shift) {
    float shiftX = cosf(angleY) * shift;
    float shiftZ = sinf(angleY) * shift;
    pos.x += shiftX;
    pos.z += shiftZ;
}