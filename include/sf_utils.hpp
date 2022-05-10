#ifndef SF_UTILS_HPP
#define SF_UTILS_HPP

#include <Eigen/Core>
#include <ctime>
#include <string>

namespace sf::utils {

struct toolchain_info_t {
    int id;
    std::string compiler;
    std::string compilervers;
    std::string libcvers;

    toolchain_info_t();
};

struct host_info_t {
    int id;
    std::string cpuname;
    std::string cpuclock;
    std::string cpuclockmax;
    std::string memclock;
    std::string L1d;
    std::string L1i;
    std::string L2;
    std::string L3;

    host_info_t();
};

struct library_info_t {
    int id;
    std::string name;
    std::string version;
};

struct timer {
    struct timespec ts;
    struct timespec tf;

    timer() { start(); }
    void start() { clock_gettime(CLOCK_MONOTONIC, &ts); }
    void stop() { clock_gettime(CLOCK_MONOTONIC, &tf); }
    double elapsed() { return (tf.tv_sec - ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec) * 1E-9; }
};

std::string exec(const char *cmd);
std::string get_alm_version();
std::string get_sleef_version();
std::string get_af_version();
std::string get_boost_version();
std::string get_gsl_version();
std::string get_sctl_version();
std::string get_baobzi_version();
std::string get_eigen_version();

template <typename VAL_T>
Eigen::VectorX<VAL_T> transform_domain(const Eigen::VectorX<VAL_T> &vals, double lower, double upper) {
    VAL_T delta = upper - lower;
    return vals.array() * delta + lower;
}

} // namespace sf::utils

#endif
