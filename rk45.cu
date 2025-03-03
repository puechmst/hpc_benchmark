#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
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

// the dimension of the state space must be small enough to fit into local registers (255).
// static definition allows the compiler to unroll loops
// test

#define STATE_DIM (10)

#define BSIZE (100)
#define NEQ (1000 * BSIZE)

struct ode_def
{
    __device__ virtual void operator()(float t, float *y, float *yp) = 0;
    __device__ __host__ virtual float getATol() = 0;
    __device__ __host__ virtual float getRTol() = 0;
};

struct my_test : public ode_def
{
    const float atol = 1e-5;
    const float rtol = 1e-2;
    __device__ void operator()(float t, float *y, float *yp)
    {
        for (int i = 0; i < STATE_DIM; i++)
            yp[i] = 1.0f + y[i] * y[i];
    }

    __device__ __host__ float getATol() { return atol; }
    __device__ __host__ float getRTol() { return rtol; }
};

template <class T>
concept OdeObject = std::is_base_of<ode_def, T>::value;

template <OdeObject T>
__global__ void rk45(T ode, float *time, float *y4, float *y5, float *step)
{
    // arrays are normally stored in registers unless STATE_DIM is too large
    // the -Xptvas -v option in CmakeLists.txt dumps true usage.
    // please check that no spill memory is used.
    float yy[STATE_DIM], cur[STATE_DIM], k1[STATE_DIM], k2[STATE_DIM], k3[STATE_DIM], k4[STATE_DIM], k5[STATE_DIM], k6[STATE_DIM];
    float h, t;
    int ide = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = ide * STATE_DIM;
    // load local data
    h = step[ide];
    t = time[ide];
    for (int i = 0; i < STATE_DIM; i++)
        yy[i] = y4[idx + i];
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
    // get new states at order 4 and 5
    for (int i = 0; i < STATE_DIM; i++)
    {
        y4[i + idx] = yy[i] + h * (B11 * k1[i] + B12 * k2[i] + B13 * k3[i] + B14 * k4[i] + B15 * k5[i]);
        y5[i + idx] = yy[i] + h * (B21 * k1[i] + B22 * k2[i] + B23 * k3[i] + B24 * k4[i] + B25 * k5[i] + B26 * k6[i]);
    }
}

void dump_properties(std::ofstream &of)
{
    // enumerare devices
    int ndevices;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&ndevices);
    for (int i = 0; i < ndevices; i++)
    {
        of << "Device " << i << ":" << std::endl;
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
    float *dy4, *dy5, *dtime, *dstep;
    my_test ode;
    thrust::host_vector<float> t(NEQ);
    thrust::host_vector<float> tf(NEQ);
    thrust::host_vector<float> y(NEQ * STATE_DIM);
    thrust::host_vector<float> y5(NEQ * STATE_DIM);
    thrust::host_vector<float> ys(NEQ * STATE_DIM);
    thrust::host_vector<float> step(NEQ);
    thrust::device_vector<float> dvt(NEQ);
    thrust::device_vector<float> dvy4(NEQ * STATE_DIM);
    thrust::device_vector<float> dvy5(NEQ * STATE_DIM);
    thrust::device_vector<float> dvstep(NEQ);
    thrust::uniform_real_distribution<float> dist(0, 0.1);
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
    thrust::generate(y.begin(), y.end(), [&]
                     { return dist(rng); });
    // save state
    ys = y;
    // set time, final time and step
    thrust::fill(t.begin(), t.end(), 0.0);
    thrust::fill(tf.begin(), tf.end(), 10.0);
    thrust::fill(step.begin(), step.end(), 1e-2);

    // iterate until final time is reached
    bool is_finished = false;
    float tpe;
    float err_level, err_estimate;
    float s;
    int nstates = 0;
    while (nstates < 100 * NEQ)
    {
        // copy state to device
        dvstep = step;
        dvy4 = ys;
        dvt = t;
        dy4 = thrust::raw_pointer_cast(&dvy4[0]);
        dy5 = thrust::raw_pointer_cast(&dvy5[0]);
        dstep = thrust::raw_pointer_cast(&dvstep[0]);
        dtime = thrust::raw_pointer_cast(&dvt[0]);
        rk45<<<nb, BSIZE>>>(ode, dtime, dy4, dy5, dstep);
        cudaDeviceSynchronize();
        // copy back from device
        y = dvy4;
        y5 = dvy5;
        for (int i = 0; i < NEQ; i++)
        {
            // check for termination
            if (t[i] >= tf[i])
            {
                // regenerate new state

                thrust::generate(&y[i * STATE_DIM], &y[(i + 1) * STATE_DIM] - 1, [&]
                                 { return dist(rng); });
                t[i] = 0.0;
                nstates++;
            }
            // error estimation
            tpe = 0.0;
            for (int j = 0; j < STATE_DIM; j++)
            {
                err_level = ode.getATol() + ode.getRTol() * y[i * STATE_DIM + j];
                err_estimate = abs(y[i * STATE_DIM + j] - y5[i * STATE_DIM + j]);
                tpe = max(tpe, err_estimate / err_level);
            }
            if (tpe >= 1.1)
            {
                // reduce step
                s = max(0.2f, 0.9 * pow(tpe, -0.25f));
            }
            else
            {
                // accept new state

                for (int j = 0; j < STATE_DIM; j++)
                    ys[i * STATE_DIM + j] = y[i * STATE_DIM + j];
                t[i] += step[i];
                // increase step
                s = min(5.0f, 0.9 * pow(tpe, -0.20f));
            }
            s *= step[i];
            step[i] = min(s, tf[i] - step[i]);
        }
    }

    res_file.close();
    return 0;
}