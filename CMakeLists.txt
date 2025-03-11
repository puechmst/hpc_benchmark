cmake_minimum_required(VERSION 3.20)
project(coursgpu LANGUAGES CXX CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
add_executable(simu rk45.cu quaternion.cuh uav.cuh ode.cuh)
set_property(TARGET simu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET simu PROPERTY CXX_STANDARD 20)
target_compile_options(simu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -std=c++20
                       -cudadevrt static
                        -Xptxas -v 
                       >)
set_target_properties(simu PROPERTIES CUDA_ARCHITECTURES "75")