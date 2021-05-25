#ifndef CUDAMAIN_H
#define CUDAMAIN_H

#include <SFML/Graphics/Color.hpp>

#include <iostream>

#include "global.h"
#include "vec3d.h"
#include "camera.h"

sf::Uint8* d_pixels;

void init() {
    cudaMalloc(&d_pixels, W * H * 4 * sizeof(sf::Uint8));
}

template<typename T>
__global__ void kernel(sf::Uint8* d_pixels, int n, T& figure, Camera* d_camera) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    while (id < n) {
        float4 color = figure.getPixel(id % W, id / W, d_camera);
        d_pixels[4 * id] = color.x;
        d_pixels[4 * id + 1] = color.y;
        d_pixels[4 * id + 2] = color.z;
        d_pixels[4 * id + 3] = color.w;
        id += gridDim.x * blockDim.x;
    }
}

template<typename T>
void update(sf::Uint8* pixels, T& figure, Camera* d_camera) {
    kernel<<<blocks, threads>>>(d_pixels, W * H, figure, d_camera);

    cudaMemcpy((void**) pixels, d_pixels, W * H * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);
}

void destruct() {
    cudaFree(d_pixels);
}

#endif