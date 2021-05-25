cmake_minimum_required(VERSION 3.8)
project(mandelbulb LANGUAGES CXX CUDA)

set(EXECUTABLE_NAME "mandelbulb")

add_library(main STATIC
    src/vec3d.cu
    src/camera.cu
    src/mandelbulb.cu
)

set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "60;72")

find_package(SFML 2 COMPONENTS graphics REQUIRED)

add_executable(${EXECUTABLE_NAME} src/main.cu)
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_ARCHITECTURES 60 72)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE main sfml-graphics)

include_directories(include)
