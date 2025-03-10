#include<cuda.h>
#include <type_traits>

#ifndef _ODE_H
#define _ODE_H

// interface for an ODE problem
struct ode_def
{
    __device__ virtual void operator()(float t, float *y, float *yp) = 0;
    __device__ __host__ virtual constexpr float getATol() = 0;
    __device__ __host__ virtual constexpr float getRTol() = 0;
    __device__ __host__ virtual constexpr int getDim() = 0;
};

template <class T>
concept OdeObject = std::is_base_of<ode_def, T>::value;

#endif