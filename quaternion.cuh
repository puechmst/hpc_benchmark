#include <cuda.h>
#include <math.h>
#include <type_traits>

namespace numeric
{
    template <typename T>
        requires(std::is_same<T, float>::value || std::is_same<T, double>::value)
    class Quaternion 
    {
    private:
        using vec4 = std::conditional_t<std::is_same<T, double>::value, double4, float4>;
        using vec3 = std::conditional_t<std::is_same<T, double>::value, double3, float3>;
        // real part is q.w.
        vec4 q;

    public:
        __host__ __device__ Quaternion() : q({0.0}) { ; }
        __host__ __device__ Quaternion(vec4 &v): q(v) {;}
        __host__ __device__ Quaternion(T x) : q({0.0, 0.0, 0.0, x}) { ; }
        __host__ __device__ Quaternion(T w, T x, T y, T z) : q({x,y,z,w}) { ; }

        __host__ __device__ operator vec4 () {
            return q;
        }

        __host__ __device__ T real() { return q.w;}
        __host__ __device__ vec3 imag() { return vec3(q.x,q.y,q.z);}

        __host__ __device__ T normSq()
        {
            return q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        }

        __host__ __device__ T norm()
        {
            return sqrt(normSq());
        }

        __host__ __device__ Quaternion operator+(Quaternion op)
        {
            return Quaternion(q.w + op.q.w, q.x + op.q.x, q.y + op.q.y, q.z + op.q.z);
        }
        __host__ __device__ Quaternion operator-(Quaternion op)
        {
            return Quaternion(q.w - op.q.w, q.x - op.q.x, q.y - op.q.y, q.z - op.q.z);
        }
        __host__ __device__ Quaternion operator*(Quaternion op)
        {
            T a0, a1, a2, a3;

            a0 = q.w * op.q.w - q.x * op.q.x - q.y * op.q.y - q.z * op.q.z;
            a1 = q.w * op.q.x + q.x * op.q.w + q.y * op.q.z - q.z * op.q.y;
            a2 = q.w * op.q.y - q.x * op.q.z + q.y * op.q.w + q.z * op.q.x;
            a3 = q.w * op.q.z + q.x * op.q.y - q.y * op.q.x + q.z * op.q.w;

            return Quaternion(a0, a1, a2, a3);
        }
        __host__ __device__ Quaternion operator/(Quaternion op)
        {
            T a0, a1, a2, a3;
            T invn = 1.0 / op.normSq();

            a0 = q.w * op.q.w + q.x * op.q.x + q.y * op.q.y + q.z * op.q.z;
            a1 = -q.w * op.q.x + q.x * op.q.w - q.y * op.q.z + q.z * op.q.y;
            a2 = -q.w * op.q.y + q.x * op.q.z + q.y * op.q.w - q.z * op.q.x;
            a3 = -q.w * op.q.z - q.x * op.q.y + q.y * op.q.x + q.z * op.q.w;

            return Quaternion(a0 * invn, a1 * invn, a2 * invn, a3 * invn);
        }
        __host__ __device__ Quaternion conj()
        {
            return Quaternion(q.w, -q.x, -q.y, -q.z);
        }
        __host__ __device__ vec3 eulerAngles()
        {
            T phi, theta, psi;

            phi = atan2(2.0 * (q.w * q.x + q.y * q.z), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z);
            theta = asin(2.0 * (q.w * q.y - q.z * q.x));
            psi = atan2(2.0 * (q.w * q.z + q.y * q.x), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z);

            return vec3(phi, theta, psi);
        }

        __host__ __device__ Quaternion derivative(vec3 omega) {
            T r0, r1, r2, r3;

            r0 = -q.x * omega.x - q.y * omega.y - q.z * omega.z;
            r1 = omega.y * q.z - omega.z * q.y + omega.x * q.w;
            r2 = omega.z * q.x - omega.x * q.z + omega.y * q.w;
            r3 = omega.x * q.y - omega.y * q.x + omega.z * q.w;

            return Quaternion(0.5 * r0, 0.5 * r1, 0.5 * r2, 0.5 * r3);
        }

        __host__ __device__ static Quaternion makeQuatFromEuler(T phi, T theta, T psi)
        {
            T r0, r1, r2, r3;
            T cphi, sphi, ctheta, stheta, cpsi, spsi;

            cphi = cos(phi * 0.5);
            sphi = sin(phi * 0.5);
            ctheta = cos(theta * 0.5);
            stheta = sin(theta * 0.5);
            cpsi = cos(psi * 0.5);
            spsi = sin(psi * 0.5);

            r0 = cphi * ctheta * cpsi + sphi * stheta * spsi;
            r1 = sphi * ctheta * cpsi - cphi * stheta * spsi;
            r2 = cphi * stheta * cpsi + sphi * ctheta * spsi;
            r3 = cphi * ctheta * spsi - sphi * stheta * cpsi;

            return Quaternion(r0, r1, r2, r3);
        }


        __host__ __device__ static Quaternion makeFromAxisAngle(T alpha, vec3 u)
        {
            T ca, sa;
            T inu = 1.0 / sqrt(u.x * u.x + u.y * u.y + u.z * u.z);

            ca = cos(alpha * 0.5);
            sa = sin(alpha * 0.5);
            inu *= sa;
            return Quaternion(ca, inu * u.x, inu * u.y, inu * u.z);
        }

        __host__ __device__ static Quaternion makeFromVectors(vec3 u, vec3 v)
        {
            T dot;
            T c2, c, s;
            T nu, nv, inw;
            vec3 w;

            w.x = u.y * v.z - u.z * v.y;
            w.y = u.z * v.x - u.x * v.z;
            w.z = u.x * v.y - u.y * v.x;

            inw = sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
            if (inw == 0.0)
            {
                return Quaternion(1.0, 0.0, 0.0, 0.0);
            }
            inw = 1.0 / inw;

            // normalize w
            w.x *= inw;
            w.y *= inw;
            w.z *= inw;

            nu = sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
            nv = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

            // compute half rotation angle sine and cosine
            dot = u.x * v.x + u.y * v.y + u.z * v.z;
            c2 = dot / (nu * nv);
            c = sqrt(0.5 * (1.0 + c2));
            s = sqrt(0.5 * (1.0 - c2));

            return Quaternion(c, w.x * s, w.y * s, w.z * s);
        }

        __host__ __device__ static Quaternion exp(Quaternion &op)
        {
            T nv = sqrt(op.q.x * op.q.x + op.q.y * op.q.y + op.q.z * op.q.z);
            T t = (nv == 0) ? 0.0 : sin(nv) / nv;
            T a = ::exp(op.q.w);
            return Quaternion(a * cos(nv), a * t * op.q.x, a * t * op.q.y , a * t * op.q.z );
        }

        __host__ __device__ static Quaternion log(Quaternion &op)
        {
            T n = op.norm();
            T nv;
            T t;
            if (n == 0.0)
            {
                return Quaternion(NAN, 0.0, 0.0, 0.0);
            }
            nv = sqrt(op.q.x * op.q.x + op.q.y * op.q.y + op.q.z * op.q.z);
            t = (nv == 0.0) ? 0.0 : acos(op.q.w / n) / nv;
            return Quaternion(::log(n), t * op.q.x, t * op.q.y, t * op.q.z);
        }


    };

};