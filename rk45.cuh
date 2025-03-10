#include "ode.cuh"


__device__ const float A11 = 1.0f / 4.0f;
__device__ const float A21 = 3.0f / 32.0f;
__device__ const float A22 = 9.0f / 32.0f;
__device__ const float A31 = 1932.0f / 2197.0f;
__device__ const float A32 = -7200.0f / 2197.0f;
__device__ const float A33 = 7296.0f / 2197.0f;
__device__ const float A41 = 439.0f / 216.0f;
__device__ const float A42 = -8.0f;
__device__ const float A43 = 3680.0f / 513.0f;
__device__ const float A44 = -845.0f / 4104.0f;
__device__ const float A51 = -8.0f / 27.0f;
__device__ const float A52 = 2.0f;
__device__ const float A53 = -3544.0f / 2565.0f;
__device__ const float A54 = 1859.0f / 4104.0f;
__device__ const float A55 = -11.0f / 40.0f;

__device__ const float B11 = 25.0f / 216.0f;
__device__ const float B12 = 0.0f;
__device__ const float B13 = 1408.0f / 2565.0f;
__device__ const float B14 = 2197.0f / 4104.0f;
__device__ const float B15 = -1.0f / 5.0f;

__device__ const float B21 = 16.0f / 135.0f;
__device__ const float B22 = 0.0f;
__device__ const float B23 = 6656.0f / 12825.0f;
__device__ const float B24 = 28561.0f / 56430.0f;
__device__ const float B25 = -9.0f / 50.0f;
__device__ const float B26 = 2.0f / 55.0f;

__device__ const float C2 = 1.0f / 4.0f;
__device__ const float C3 = 3.0f / 8.0f;
__device__ const float C4 = 12.0f / 13.0f;
__device__ const float C5 = 1.0f;
__device__ const float C6 = 1.0f / 2.0f;



template <OdeObject T>
__global__ void rk45(T ode, float *time, float *y4, float *y5, float *step)
{
    // arrays are normally stored in registers unless ode.getDim() is too large
    // the -Xptvas -v option in CmakeLists.txt dumps true usage.
    // please check that no spill memory is used.
    float yy[ode.getDim()], cur[ode.getDim()], k1[ode.getDim()], k2[ode.getDim()], k3[ode.getDim()], k4[ode.getDim()], k5[ode.getDim()], k6[ode.getDim()];
    float h, t;
    int ide = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = ide * ode.getDim();
    // load local data
    h = step[ide];
    t = time[ide];
    for (int i = 0; i < ode.getDim(); i++)
        yy[i] = y4[idx + i];
    ode(t, yy, k1);
    for (int i = 0; i < ode.getDim(); i++)
        cur[i] = yy[i] + h * A11 * k1[i];
    ode(t + h * C2, cur, k2);
    for (int i = 0; i < ode.getDim(); i++)
        cur[i] = yy[i] + h * (A21 * k1[i] + A22 * k2[i]);
    ode(t + h * C3, cur, k3);
    for (int i = 0; i < ode.getDim(); i++)
        cur[i] = yy[i] + h * (A31 * k1[i] + A32 * k2[i] + A33 * k3[i]);
    ode(t + h * C4, cur, k4);
    for (int i = 0; i < ode.getDim(); i++)
        cur[i] = yy[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i] + A44 * k4[i]);
    ode(t + h * C5, cur, k5);
    for (int i = 0; i < ode.getDim(); i++)
        cur[i] = yy[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i] + A55 * k5[i]);
    ode(t + h * C6, cur, k6);
    // get new states at order 4 and 5
    for (int i = 0; i < ode.getDim(); i++)
    {
        y4[i + idx] = yy[i] + h * (B11 * k1[i] + B12 * k2[i] + B13 * k3[i] + B14 * k4[i] + B15 * k5[i]);
        y5[i + idx] = yy[i] + h * (B21 * k1[i] + B22 * k2[i] + B23 * k3[i] + B24 * k4[i] + B25 * k5[i] + B26 * k6[i]);
    }
}