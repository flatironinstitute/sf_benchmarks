#ifndef SF_UTILS_HPP
#define SF_UTILS_HPP

#include <string>

namespace sf::utils {

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
