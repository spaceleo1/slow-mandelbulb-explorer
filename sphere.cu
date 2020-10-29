#include "sphere.h"

Sphere::Sphere(float x, float y, float z, float r) : 
    c(Vec3d(x, y, z)), r(r) {}
Sphere::~Sphere() {}


__host__ __device__ float Sphere::getDist(const Vec3d& c, float r, const Vec3d& v) {
    return (v - c).abs() - r;
}

__host__ __device__ Vec3d Sphere::getNormal(const Vec3d& c, float r, const Vec3d& v) {
    float d = getDist(c, r, v);
    
    const float eps = 0.001f;
    const Vec3d v1(eps, 0, 0);
    const Vec3d v2(0, eps, 0);
    const Vec3d v3(0, 0, eps);
    
    Vec3d norm(getDist(c, r, v - v1),
               getDist(c, r, v - v2),
               getDist(c, r, v - v3));
    norm = Vec3d(d, d, d) - norm;
    norm.normalize();

    return norm;
}

__host__ __device__ float4 Sphere::getPixel(const Vec3d& c, float r, int x, int y, const Camera& camera) {
    Vec3d lightPos(0, -1, 0);
    Vec3d dir((float(x) / float(W) * 2.0f - 1.0f) * float(W) / float(H), float(y) / float(H) * 2.0f - 1.0f, camera.f);

    float newX = dir.x * cosf(camera.angleY) - dir.z * sinf(camera.angleY);
    float newZ = dir.x * sinf(camera.angleY) + dir.z * cosf(camera.angleY);

    dir.x = newX;
    dir.z = newZ;

    dir.normalize();

    float t = 0;
    
    int it = 0;
    for (; it < MAX_RAYMARCH_ITERS; ++it) {
        Vec3d p = camera.pos + dir * t;
        float d = getDist(c, r, p);

        if (d < 0.001) {
            Vec3d v = getNormal(c, r, p);
            float w = v * lightPos;
            float color = (w + 1.0f) / 2.0f * 255.0f;
            float s = sinf(p.z * acosf(-1) * 3.0f / 2.0f);

            return make_float4(color, color * ((s + 1.0f) / 2.0f), color * ((s + 1.0f) / 2.0f), 255);
        }

        t += d;
    }

    return make_float4(255, 255, 255, 255);
}
