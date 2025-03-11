#include "quaternion.cuh"
#include "ode.cuh"
#include "helper_math.h"

using namespace numeric;

// the dynamic equation of a UAV
struct uav_dynamics : public ode_def
{
    static constexpr float atol = 1e-5;
    static constexpr float rtol = 1e-2;
    static constexpr int dim = 12;
    static constexpr int POS_X = 0;
    static constexpr int POS_Y = 1;
    static constexpr int POS_Z = 2;

    // spinor attitude component (lg of the real attitude)
    static constexpr int R0 = 3;
    static constexpr int R1 = 4;
    static constexpr int R2 = 5;

    // derivatives of positions
    static constexpr int DX = 6;
    static constexpr int DY = 7;
    static constexpr int DZ = 8;

    // angular velocities
    static constexpr int OMEGA1 = 9;
    static constexpr int OMEGA2 = 10;
    static constexpr int OMEGA3 = 11;
    // mass
    static constexpr float uav_mass = 2.0;
    // force gain
    static constexpr float f_gain = 0.1;
    // attitude gains
    static constexpr float attitude_gain= 0.1;
    // typical inertia
    static constexpr float3 inertia = {0.008513, 0.008513, 0.015579};
    // drag
    static constexpr float dragCoefficient = 1e-2;
    // gravitation constant
    static constexpr float g = 9.81;
    // weather data is coded according to this scheme:
    // x -> temperature in kelvins
    // y -> east component of wind
    // z -> north component of wind
    // w -> geoaltitude
    cudaTextureObject_t tex;
    // target position and velocity.
    // x,y,z have their usual meanings (in meters)
    // w is the target velocity in meters per second
    float4 *target;
    __device__ void operator()(float t, float *y, float *yp)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        float4 wd = tex3D<float4>(tex, 0.1, 0.1, 0.1);
        float3 pos_err = make_float3({target[id].x-y[POS_X]}, target[id].y-y[POS_Y], target[id].z-y[POS_Z]);
        float v_err = target[id].w - length(make_float3(y[DX]+wd.x, y[DY]+wd.y, y[DZ]+wd.z));
        // compute force
        float f = f_gain *  v_err / uav_mass;
        // compute attitude error
        auto target_attitude = Quaternion<float>::log(Quaternion<float>::makeFromVectors(pos_err, float3(1.0,0.0,0.0)));
        auto err_attitude = target_attitude - Quaternion<float>(0.0, y[R0], y[R1], y[R2]);
        float3 cmd;
        cmd.x = -attitude_gain * err_attitude.getX() - attitude_gain * y[OMEGA1];
        cmd.y = -attitude_gain * err_attitude.getY() -attitude_gain * y[OMEGA2];
        cmd.z = -attitude_gain * err_attitude.getZ() - attitude_gain * y[OMEGA3];
        // apply command
        yp[OMEGA1] = cmd.x * inertia.x;
        yp[OMEGA2] = cmd.y * inertia.y;
        yp[OMEGA3] = cmd.z * inertia.z;
        float r = length(make_float3(y[R0],y[R1],y[R2]));
        float fac = 1.0 - M_PI / r;
        if (r >= M_PI) {
            y[R0] *= fac;
            y[R1] *= fac;
            y[R2] *= fac;
        }
        float a,o1, o2, o3, r1, r2, r3, ro1, ro2, ro3;
        o1 = y[OMEGA1];
        o2 = y[OMEGA2];
        o3 = y[OMEGA3];

        r1 = y[R0];
        r2 = y[R1];
        r3 = y[R2];

        ro1 = 0.5 * (o2 * r3 - o3 * r2);
        ro2 = 0.5 * (o3 * r1 - o1 * r3);
        ro3 = 0.5 * (o1 * r2 - o2 * r1);
        // common factor
        if (r == 0.0)
            a = 1.0;
        else
            a = r / tan(r);
        // angular velocities contribution
        ro1 += 0.5 * a * o1;
        ro2 += 0.5 * a * o2;
        ro3 += 0.5 * a * o3;
        // spinor contribution
        if (r > 0.0) {
            double dot = r1 * o1 + r2 * o2 + r3 * o3;
            double b = 0.5 * dot * (1.0 - a) / (r * r);
            ro1 += r1 * b;
            ro2 += r2 * b;
            ro3 += r3 * b;
        }
        yp[R0] = ro1;
        yp[R1] = ro2;
        yp[R2] = ro3;
        Quaternion<float> attitude = Quaternion<float>::exp(Quaternion<float>(0.0, r1, r2, r3));
          // compute total force in earth frame
          r1 = f * 2.0 * (attitude.getX() * attitude.getZ() - attitude.real() * attitude.getY());
          r2 = f * 2.0 * (attitude.getY() * attitude.getZ() + attitude.real() * attitude.getX());
          r3 = f * (attitude.getX() * attitude.real() - attitude.getX() * attitude.getX()
                  - attitude.getY() * attitude.getY() + attitude.getZ() * attitude.getZ());
          // total drag
          double ny = sqrt(y[DX] * y[DX] + y[DY] * y[DY] + y[DZ] * y[DZ]);
          float d1, d2, d3;
          d1 = -y[DX] * dragCoefficient * ny;
          d2 = -y[DY] * dragCoefficient * ny;
          d3 = -y[DZ] * dragCoefficient * ny;
          // compute accelerations
          yp[DX] = (r1 + d1) / uav_mass;
          yp[DY] = (r2 + d2) / uav_mass;
          yp[DZ] = - g + (r3 + d3) / uav_mass;
          // compute speeds
          yp[POS_X] = y[DX];
          yp[POS_Y] = y[DY];
          yp[POS_Z] = y[DZ];
    }

    __device__ __host__ constexpr float getATol() { return atol; }
    __device__ __host__ constexpr float getRTol() { return rtol; }
    __device__ __host__ constexpr int getDim() { return dim; }
};