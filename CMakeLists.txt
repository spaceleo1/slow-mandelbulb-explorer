cmake_minimum_required(VERSION 3.8)
project(mandelbulb LANGUAGES CXX CUDA)

set(EXECUTABLE_NAME "mandelbulb")

add_library(main STATIC
    sphere.cu
    sphere.h
    vec3d.cu
    vec3d.h
    camera.cu
    camera.h
    mandelbulb.cu
    mandelbulb.h
)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(SFML 2 COMPONENTS graphics REQUIRED)

add_executable(${EXECUTABLE_NAME} main.cu)
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE main sfml-graphics)
