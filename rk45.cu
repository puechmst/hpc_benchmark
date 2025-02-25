#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define A11 (1/4)
#define A21 (3/32)
#define A22 (9/32)
#define A31 (1932/2197)
#define A32 (-7200/2197)
#define A33 (7296/2197)
#define A41 (439/216)
#define A42 (-8)
#define A43 (3680/513)
#define A44 (-845/4104)
#define A51 (-8/27)
#define A52 (2)
#define A53 (-3544/2565)
#define A54 (1859/4104)
#define A55 (-11/40)

#define B11 (25/216)
#define B12  (0)
#define B13 (1408/2565)
#define B14 (2197/4101)
#define B15 (-1/5)

#define B21 (16/135)
#define B22  (0)
#define B23 (6656/12825)
#define B24 (28561/56430)
#define B25 (-9/50)
#define B26 (2/55)

#define C2 (1/4)
#define C3 (3/8)
#define C4 (12/13)
#define C5 (1)
#define C6 (1/2)


__device__ void sysdyn(float t, int n, float *y, float *yp) {
    // solution: y = tan(t)
    for(int i = 0 ; i < n ; i++)
        yp[i] = 1.0f + y[i];
}

__global__ void rk45(float t, int n, float *y, float step) {
    float k1,k2,k3,k4,k5,k6;
    float h = step;
    float z4,z5;

    k1 = sysdyn(t,y);
    k2 = sysdyn(t+C2,y + A11 * k1);
    k3 = sysdyn(t+C3, y + A21 * k1 + A22 * k2);
    k4 = sysdyn(t+C4, y + A31 * k1 + A32 * k2 + A33 * k3);
    k5 = sysdyn(t+C4, y + A41 * k1 + A42 * k2 + A43 * k3 + A44 * k4);
    k6 = sysdyn(t+C6, y + A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4 + A55 * k5);

}