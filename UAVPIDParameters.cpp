#include <array>

class UAVPIDParameters {
    private:
        typedef std::array<double, 6> t_gains;
        typedef std::array<double, 4> t_limits;
        t_gains attitudeGains;
        t_gains navigationGains;
        t_limits limits;
    public:
        UAVPIDParameters(t_gains attitude, t_gains nav, t_limits lim) : attitudeGains(attitude), navigationGains(nav), limits(lim) {;}
        
}