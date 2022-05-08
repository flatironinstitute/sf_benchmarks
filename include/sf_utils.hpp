#ifndef SF_UTILS_HPP
#define SF_UTILS_HPP

#include <string>

namespace sf::utils {

struct toolchain_info_t {
    std::string compiler;
    std::string compilervers;
    std::string libcvers;

    toolchain_info_t();
};

struct host_info_t {
    std::string cpu_name;
    std::string L1d;
    std::string L1i;
    std::string L2;
    std::string L3;

    host_info_t();
};

struct library_info_t {
    std::string name;
    std::string version;
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

} // namespace sf::utils

#endif
