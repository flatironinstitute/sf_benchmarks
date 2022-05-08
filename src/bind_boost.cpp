#include <dlfcn.h>
#include <string>
#include <unordered_map>

#include <sf_libraries.hpp>

namespace sf::functions::boost {
std::unordered_map<std::string, multi_eval_func<float>> funs_fx1 = {
    {"sin_pi", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sin_pi(x); })},
    {"cos_pi", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cos_pi(x); })},
    {"tgamma", scalar_func_apply<float>([](float x) -> float { return ::boost::math::tgamma<float>(x); })},
    {"lgamma", scalar_func_apply<float>([](float x) -> float { return ::boost::math::lgamma<float>(x); })},
    {"digamma", scalar_func_apply<float>([](float x) -> float { return ::boost::math::digamma<float>(x); })},
    {"pow13", scalar_func_apply<float>([](float x) -> float { return ::boost::math::pow<13>(x); })},
    {"erf", scalar_func_apply<float>([](float x) -> float { return ::boost::math::erf(x); })},
    {"erfc", scalar_func_apply<float>([](float x) -> float { return ::boost::math::erfc(x); })},
    {"sinc_pi", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sinc_pi(x); })},
    {"bessel_Y0", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_neumann(0, x); })},
    {"bessel_Y1", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_neumann(1, x); })},
    {"bessel_Y2", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_neumann(2, x); })},
    {"bessel_I0", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_i(0, x); })},
    {"bessel_I1", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_i(1, x); })},
    {"bessel_I2", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_i(2, x); })},
    {"bessel_J0", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_j(0, x); })},
    {"bessel_J1", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_j(1, x); })},
    {"bessel_J2", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_j(2, x); })},
    {"bessel_K0", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_k(0, x); })},
    {"bessel_K1", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_k(1, x); })},
    {"bessel_K2", scalar_func_apply<float>([](float x) -> float { return ::boost::math::cyl_bessel_k(2, x); })},
    {"bessel_j0", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sph_bessel(0, x); })},
    {"bessel_j1", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sph_bessel(1, x); })},
    {"bessel_j2", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sph_bessel(2, x); })},
    {"bessel_y0", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sph_neumann(0, x); })},
    {"bessel_y1", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sph_neumann(1, x); })},
    {"bessel_y2", scalar_func_apply<float>([](float x) -> float { return ::boost::math::sph_neumann(2, x); })},
    {"hermite_0", scalar_func_apply<float>([](float x) -> float { return ::boost::math::hermite(0, x); })},
    {"hermite_1", scalar_func_apply<float>([](float x) -> float { return ::boost::math::hermite(1, x); })},
    {"hermite_2", scalar_func_apply<float>([](float x) -> float { return ::boost::math::hermite(2, x); })},
    {"hermite_3", scalar_func_apply<float>([](float x) -> float { return ::boost::math::hermite(3, x); })},
    {"riemann_zeta", scalar_func_apply<float>([](float x) -> float { return ::boost::math::zeta(x); })},
};

std::unordered_map<std::string, multi_eval_func<double>> funs_dx1 = {
    {"sin_pi", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sin_pi(x); })},
    {"cos_pi", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cos_pi(x); })},
    {"tgamma", scalar_func_apply<double>([](double x) -> double { return ::boost::math::tgamma<double>(x); })},
    {"lgamma", scalar_func_apply<double>([](double x) -> double { return ::boost::math::lgamma<double>(x); })},
    {"digamma", scalar_func_apply<double>([](double x) -> double { return ::boost::math::digamma<double>(x); })},
    {"pow13", scalar_func_apply<double>([](double x) -> double { return ::boost::math::pow<13>(x); })},
    {"erf", scalar_func_apply<double>([](double x) -> double { return ::boost::math::erf(x); })},
    {"erfc", scalar_func_apply<double>([](double x) -> double { return ::boost::math::erfc(x); })},
    {"sinc_pi", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sinc_pi(x); })},
    {"bessel_Y0", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_neumann(0, x); })},
    {"bessel_Y1", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_neumann(1, x); })},
    {"bessel_Y2", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_neumann(2, x); })},
    {"bessel_I0", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_i(0, x); })},
    {"bessel_I1", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_i(1, x); })},
    {"bessel_I2", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_i(2, x); })},
    {"bessel_J0", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_j(0, x); })},
    {"bessel_J1", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_j(1, x); })},
    {"bessel_J2", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_j(2, x); })},
    {"bessel_K0", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_k(0, x); })},
    {"bessel_K1", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_k(1, x); })},
    {"bessel_K2", scalar_func_apply<double>([](double x) -> double { return ::boost::math::cyl_bessel_k(2, x); })},
    {"bessel_j0", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sph_bessel(0, x); })},
    {"bessel_j1", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sph_bessel(1, x); })},
    {"bessel_j2", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sph_bessel(2, x); })},
    {"bessel_y0", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sph_neumann(0, x); })},
    {"bessel_y1", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sph_neumann(1, x); })},
    {"bessel_y2", scalar_func_apply<double>([](double x) -> double { return ::boost::math::sph_neumann(2, x); })},
    {"hermite_0", scalar_func_apply<double>([](double x) -> double { return ::boost::math::hermite(0, x); })},
    {"hermite_1", scalar_func_apply<double>([](double x) -> double { return ::boost::math::hermite(1, x); })},
    {"hermite_2", scalar_func_apply<double>([](double x) -> double { return ::boost::math::hermite(2, x); })},
    {"hermite_3", scalar_func_apply<double>([](double x) -> double { return ::boost::math::hermite(3, x); })},
    {"riemann_zeta", scalar_func_apply<double>([](double x) -> double { return ::boost::math::zeta(x); })},
};

std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1() { return funs_fx1; }
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1() { return funs_dx1; }
} // namespace sf::functions::boost
