#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <texture_types.h>
#include <iostream>
#include <fstream>
#include <type_traits>
#include "rk45.cuh"
#include "uav.cuh"


// the dimension of the state space must be small enough to fit into local registers (255).
// static definition allows the compiler to unroll loops
// test

#define STATE_DIM (12)

#define BSIZE (100)
#define NEQ (10000 * BSIZE)


// the dynamic equation of a UAV
struct my_test : public ode_def
{
    static constexpr float atol = 1e-5;
    static constexpr float rtol = 1e-2;
    static constexpr int dim = STATE_DIM;
    cudaTextureObject_t tex;
    __device__ void operator()(float t, float *y, float *yp)
    {
        float4 wd = tex3D<float4>(tex, 0.1, 0.1, 0.1);
     
        for (int i = 0; i < STATE_DIM; i++)
            yp[i] = 1.0f + y[i] * y[i] + 0.01 * wd.y;
    }

    __device__ __host__ constexpr float getATol() { return atol; }
    __device__ __host__ constexpr float getRTol() { return rtol; }
    __device__ __host__ constexpr int  getDim() { return dim; }
};



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
        of << "Texture size: " << prop.maxTexture3D[0] << "x" << prop.maxTexture3D[1] << "x" << prop.maxTexture3D[1] << std::endl;
        of << "shared memory (per block) : " << prop.sharedMemPerBlock << std::endl;
        of << "registers (per block) : " << prop.regsPerBlock << std::endl;
        of << "registers (per mp) : " << prop.regsPerMultiprocessor << std::endl;
    }
}

/**
 * @brief This function generates a synthetic weather grid with the format of ERA5 data
 * 
 * @param dst 
 * @param nlon 
 * @param nlat 
 * @param nlevel 
 */
void generateWeatherData(float4 *dst, int nlon, int nlat, int nlevel) {
    int idx = 0;
    float lat, lon, h, p;
    // the wind has a negative gradient, going to zero at the poles
    for(int ih = 0 ; ih < nlevel ; ih++) {
        for(int ilat = 0 ; ilat < nlat ; ilat++) {
            // wind is constant in altitude
            // temperature is ISA
            for(int ilon = 0 ; ilon < nlon ; ilon++) {
                lat = (float)(ilat-nlat/2) * 0.25; 
                lon = (float)(ilon-nlon/2) * 0.25;
                p = 1000.0f - (float)ih * 50.0f;
                // barometric equation
                h = 44307 * (1 - pow(p/1013, 0.19));
                // temperature in kelvins
                dst[idx].x = 288 - 0.006 * h;
                // east component of wind
                dst[idx].y = 1.0f * (1 - abs(lat)/ 90.0f);
                // north component of wind
                dst[idx].z = 0.0f;
                // geoaltitude
                dst[idx++].w = h;
            }
        }
    }
}


int main(int argc, char *argv[])
{

    // device raw pointers
    float *dy4, *dy5, *dtime, *dstep;
    cudaTextureObject_t weatherTex;
    // format is altitude, temperature, vx, vy, hence 4 floats
    cudaChannelFormatDesc weatherDesc = cudaCreateChannelDesc<float4>();
    cudaArray *weatherArray;
    float4 *syntheticWeather;
    cudaError_t err;

    // weather grid has 0.25° resolution
    constexpr int nlat = 4 * 180;
    constexpr int nlon = 4 * 360;
    // number of pressure levels
    constexpr int nlevel = 17;
    // create device array
    cudaMalloc3DArray(&weatherArray, &weatherDesc, make_cudaExtent(nlon, nlat, nlevel));
    // create host array 
    syntheticWeather = new float4[nlat * nlon * nlevel ];
    generateWeatherData(syntheticWeather, nlon, nlat, nlevel);
    // copy parameters
    cudaMemcpy3DParms cpyparms = {0};
    cpyparms.srcPos = make_cudaPos(0,0,0);
    cpyparms.dstPos = make_cudaPos(0,0,0);
    cpyparms.srcPtr = make_cudaPitchedPtr(syntheticWeather,  nlon * sizeof(float4), nlon, nlat);
    cpyparms.dstArray = weatherArray;
    cpyparms.extent = make_cudaExtent(nlon, nlat, nlevel);
    cpyparms.kind = cudaMemcpyHostToDevice;
    // copy array
    cudaMemcpy3D(&cpyparms);
    // create texture
    cudaResourceDesc rd;
    memset(&rd, 0, sizeof(cudaResourceDesc));
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = weatherArray;
    cudaTextureDesc td;
    memset(&td, 0, sizeof(cudaTextureDesc));
    td.addressMode[0] = cudaAddressModeWrap;
    td.addressMode[1] = cudaAddressModeWrap;
    td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    td.normalizedCoords = true;
    cudaCreateTextureObject(&weatherTex, &rd, &td, nullptr);
    // the problem to be solved
    struct uav_dynamics ode;
    ode.tex = weatherTex;
    // current time vector
    thrust::host_vector<float> t(NEQ);
    // final time vector
    thrust::host_vector<float> tf(NEQ);
    // low order state
    thrust::host_vector<float> y(NEQ * STATE_DIM);
    // high order state
    thrust::host_vector<float> y5(NEQ * STATE_DIM);
    // saved state
    thrust::host_vector<float> ys(NEQ * STATE_DIM);
    // current steo
    thrust::host_vector<float> step(NEQ);
    // device current time vector
    thrust::device_vector<float> dvt(NEQ);
    // device low order state
    thrust::device_vector<float> dvy4(NEQ * STATE_DIM);
    // device high order state
    thrust::device_vector<float> dvy5(NEQ * STATE_DIM);
    // device step
    thrust::device_vector<float> dvstep(NEQ);
    // device target
    thrust::host_vector<float4> target(NEQ);
    thrust::device_vector<float4> dtarget(NEQ);
    // random number generator stuff
    thrust::uniform_real_distribution<float> dist(0, 0.1);
    thrust::default_random_engine rng(1234);

    // number of blocks
    int nb = (NEQ + BSIZE - 1) / BSIZE;


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
    // populate target
    thrust::generate(target.begin(),target.end(), [&] { return make_float4(10000.0, 10000.0, 100.0, 1.0);});
    dtarget = target;
    ode.target = thrust::raw_pointer_cast(&dtarget[0]);
    // save state
    ys = y;
    // set time, final time and step
    thrust::fill(t.begin(), t.end(), 0.0);
    thrust::fill(tf.begin(), tf.end(), 10.0);
    thrust::fill(step.begin(), step.end(), 1e-2);

    // iterate until final time is reached
    float tpe;
    float err_level, err_estimate;
    float s;
    int nstates = 0;
    while (nstates < 10 * NEQ)
    {
        // copy state to device
        dvstep = step;
        dvy4 = ys;
        dvt = t;
        // convert thrust vectors to device pointers
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
    cudaDestroyTextureObject(weatherTex);
    cudaFreeArray(weatherArray);
    res_file.close();
    return 0;
}