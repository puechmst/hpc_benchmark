#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <iostream>
#include <fstream>
#include <type_traits>

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
__device__ const float B14 = 2197.0f / 4101.0f;
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

// the dimension of the state space must be small enough to fit into local registers (255).
// static definition allows the compiler to unroll loops
// test

#define STATE_DIM (10)

#define BSIZE (200)
#define NEQ (1000 * BSIZE)


// __device__ void sysdyn(float t, float *y, float *yp)
// {
//     // solution: y = tan(t)
//     for (int i = 0; i < STATE_DIM; i++)
//         yp[i] = 1.0f + y[i] * y[i];
// }

struct ode_def {
     __device__ virtual void operator()(float t, float *y, float *yp) = 0;
     __device__ virtual float getTol() = 0;
};

struct my_test: public ode_def {
    const float tol = 1e-5;
    __device__ void operator()(float t, float *y, float *yp) {
        for (int i = 0; i < STATE_DIM; i++)
            yp[i] = 1.0f + y[i] * y[i];
    }

    __device__ float getTol() { return tol; }
};

template<class T>
concept OdeObject = std::is_base_of<ode_def, T>::value;

template<OdeObject T>
__global__ 
void rk45(T ode, float t, float *y, float *err, float *step)
{
    // arrays are normally stored in registers unless STATE_DIM is too large
    // the -Xptvas -v option in CmakeLists.txt dumps true usage.
    float yy[STATE_DIM], cur[STATE_DIM], k1[STATE_DIM], k2[STATE_DIM], k3[STATE_DIM], k4[STATE_DIM], k5[STATE_DIM], k6[STATE_DIM];
    float e;
    float h;
    int ide = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = ide * STATE_DIM;
    // load local data
    h = step[ide];
    for (int i = 0; i < STATE_DIM; i++)
        yy[i] = y[idx + i];
    ode(t, yy, k1);
    for (int i = 0; i < STATE_DIM; i++)
        cur[i] = yy[i] + h * A11 * k1[i];
    ode(t + h * C2, cur, k2);
    for (int i = 0; i < STATE_DIM; i++)
        cur[i] = yy[i] + h * (A21 * k1[i] + A22 * k2[i]);
    ode(t + h * C3, cur, k3);
    for (int i = 0; i < STATE_DIM; i++)
        cur[i] = yy[i] + h * (A31 * k1[i] + A32 * k2[i] + A33 * k3[i]);
    ode(t + h * C4, cur, k4);
    for (int i = 0; i < STATE_DIM; i++)
        cur[i] = yy[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i] + A44 * k4[i]);
    ode(t + h * C5, cur, k5);
    for (int i = 0; i < STATE_DIM; i++)
        cur[i] = yy[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i] + A55 * k5[i]);
    ode(t + h * C6, cur, k6);
    // get new state and estimate error
    e = 0.0;
    for (int i = 0; i < STATE_DIM; i++)
    {
        // It is tempting to use the higher order approximation, but the predicted error is computed for the lower one,
        // and so is the optimal h.
        y[i + idx] = yy[i] + h * (B11 * k1[i] + B12 * k2[i] + B13 * k3[i] + B14 * k4[i] + B15 * k5[i]);
        e += h * fabs((B11 - B21) * k1[i] + (B12 - B22) * k2[i] + (B13 - B23) * k3[i] + (B14 - B24) * k4[i] + (B15 - B25) * k5[i] - B26 * k6[i]);
    }
    // save error
    err[ide] = e;
    // save optimal step for tolerance
    step[ide] =  h * 0.84 * pow( (float)STATE_DIM * ode.getTol() / e , 0.25f);
}

void dump_properties(std::ofstream &of) {
    // enumerare devices
    int ndevices;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&ndevices);
    for(int i = 0 ; i < ndevices ; i++) {
        of << "Device " << i <<  ":" << std::endl;
        cudaGetDeviceProperties(&prop, i);
        of << "name : " << prop.name << std::endl;
        of << "arch : " << prop.major << "." << prop.minor << std::endl;
        of << "global memory : " << prop.totalGlobalMem << std::endl;
        of << "shared memory (per block) : " << prop.sharedMemPerBlock << std::endl;
        of << "registers (per block) : " << prop.regsPerBlock << std::endl;
        of << "registers (per mp) : " << prop.regsPerMultiprocessor << std::endl;
    }
}

int main(int argc, char *argv[])
{
    float *dy, *derr, *dstep;
    thrust::host_vector<float> y(NEQ * STATE_DIM);
    thrust::host_vector<float> ys(NEQ * STATE_DIM);
    thrust::host_vector<float> err(NEQ);
    thrust::host_vector<float> step(NEQ);
    thrust::host_vector<float> istep(NEQ);
    thrust::device_vector<float> dvy(NEQ * STATE_DIM);
    thrust::device_vector<float> dverr(NEQ);
    thrust::device_vector<float> dvstep(NEQ);
    thrust::uniform_real_distribution<float> dist(0,0.1);
    thrust::default_random_engine rng(1234);

    int nb = (NEQ + BSIZE - 1) / BSIZE;
 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // file for saving results
    std::ofstream res_file("res.txt");
    // dump capabilities
    dump_properties(res_file);
    res_file << "neq : " << NEQ << std::endl;
    res_file << "dim : " << STATE_DIM << std::endl;
    res_file << "nb : " << nb << std::endl;
    // populate state randomly
    thrust::generate(y.begin(),y.end(),[&] { return dist(rng);});
    // save state
    ys = y;
    // set step
    thrust::fill(step.begin(), step.end(), 1e-2);
    dvstep = step;
    istep = step;
    // copy state to device
    dvy = y;
    dy = thrust::raw_pointer_cast(&dvy[0]);
    derr = thrust::raw_pointer_cast(&dverr[0]);
    dstep = thrust::raw_pointer_cast(&dvstep[0]);
    cudaEventRecord(start);
    rk45<<<nb, BSIZE>>>(my_test(), 0.0, dy, derr, dstep);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millis=0;
    cudaEventElapsedTime(&millis, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    res_file << "Elapsed : " << millis << std::endl;
    float yt;
    float te;
    float max_err = 0.0;
    float max_pred_err = 0.0;
    // copy back from device
    step = dvstep;
    y = dvy;
    err = dverr;
    for (int i = 0; i < NEQ; i++)
    {
        std::cout << step[i] << std::endl;
        te = 0.0;
        for (int j = 0; j < STATE_DIM; j++)
        {
            yt = tan(istep[i] + atan(ys[i * STATE_DIM + j]));
            te += abs(yt - y[i * STATE_DIM + j]);
           
        }
        if (te > max_err)
                max_err = te;
        if (abs(err[i]) > max_pred_err)
            max_pred_err = abs(err[i]);
    }
    std::cout << "max error: " << max_err << " max predicted error: " << max_pred_err << std::endl;
    res_file.close();
    return 0;
}