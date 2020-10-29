#include "mandelbulb.h"

Mandelbulb::Mandelbulb() {}
Mandelbulb::~Mandelbulb() {}


__host__ __device__ float Mandelbulb::getDist(const Vec3d& v) {
    Vec3d z = v;
    float dr = 1.0;
    float r = 0.0;

    int it = 0;
    for (; it < MAX_MANDELBULB_ITERS; ++it) {
        r = z.abs();
        if (r > MANDELBULB_ESCAPE_RADIUS) {
            break;
        }

        float theta = acosf(z.z / r);
        float phi = atan2f(z.y, z.x);

        float zr = powf(r, 8);
        dr = zr / r * MANDELBULB_POWER * dr + 1.0;

        theta = theta * MANDELBULB_POWER;
        phi = phi * MANDELBULB_POWER;

        z = Vec3d(sinf(theta) * cosf(phi), sinf(phi) * sinf(theta), cos(theta)) * zr;
        z = z + v;
    }

    return 0.5 * logf(r) * r / dr;
}

__host__ __device__ Vec3d Mandelbulb::getNormal(const Vec3d& v, float d) {
    const float eps = 0.001f;
    const Vec3d v1(eps, 0, 0);
    const Vec3d v2(0, eps, 0);
    const Vec3d v3(0, 0, eps);
    
    Vec3d norm(getDist(v - v1),
               getDist(v - v2),
               getDist(v - v3));
    norm = Vec3d(d, d, d) - norm;
    norm.normalize();
    
    return norm;
}

__host__ __device__ float4 Mandelbulb::getPixel(int x, int y, const Camera& camera) {
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
        float d = getDist(p);

        if (d < 0.001) {
            Vec3d v = getNormal(p, d);
            float w = v * lightPos;
            float color = (w + 1.0f) / 2.0f * 255.0f;

            return make_float4(color, color, color, 255);
        }

        t += d;
    }

    return make_float4(255, 255, 255, 255);
}
