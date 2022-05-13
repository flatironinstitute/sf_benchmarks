#include <sf_libraries.hpp>

namespace sf::functions::gsl {
std::unordered_map<std::string, multi_eval_func<double>> funs_dx1;
std::unordered_map<std::string, multi_eval_func<cdouble>> funs_cdx1;
bool initialized = false;

inline cdouble gsl_complex_wrapper(cdouble z, int (*f)(double, double, gsl_sf_result *, gsl_sf_result *)) {
    gsl_sf_result re, im;
    f(z.real(), z.imag(), &re, &im);
    return cdouble{re.val, im.val};
}

void load_functions() {
    if (initialized)
        return;
    initialized = true;

    funs_dx1 = {
        {"sin_pi", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sin_pi(x); })},
        {"cos_pi", scalar_func_apply<double>([](double x) -> double { return gsl_sf_cos_pi(x); })},
        {"sin", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sin(x); })},
        {"cos", scalar_func_apply<double>([](double x) -> double { return gsl_sf_cos(x); })},
        {"sinc", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sinc(x / M_PI); })},
        {"sinc_pi", scalar_func_apply<double>([](double x) -> double { return gsl_sf_sinc(x); })},
        {"erf", scalar_func_apply<double>([](double x) -> double { return gsl_sf_erf(x); })},
        {"erfc", scalar_func_apply<double>([](double x) -> double { return gsl_sf_erfc(x); })},
        {"tgamma", scalar_func_apply<double>([](double x) -> double { return gsl_sf_gamma(x); })},
        {"lgamma", scalar_func_apply<double>([](double x) -> double { return gsl_sf_lngamma(x); })},
        {"log", scalar_func_apply<double>([](double x) -> double { return gsl_sf_log(x); })},
        {"exp", scalar_func_apply<double>([](double x) -> double { return gsl_sf_exp(x); })},
        {"pow13", scalar_func_apply<double>([](double x) -> double { return gsl_sf_pow_int(x, 13); })},
        {"bessel_Y0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Y0(x); })},
        {"bessel_Y1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Y1(x); })},
        {"bessel_Y2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Yn(2, x); })},
        {"bessel_I0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_I0(x); })},
        {"bessel_I1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_I1(x); })},
        {"bessel_I2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_In(2, x); })},
        {"bessel_J0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_J0(x); })},
        {"bessel_J1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_J1(x); })},
        {"bessel_J2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Jn(2, x); })},
        {"bessel_K0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_K0(x); })},
        {"bessel_K1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_K1(x); })},
        {"bessel_K2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_Kn(2, x); })},
        {"bessel_j0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_j0(x); })},
        {"bessel_j1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_j1(x); })},
        {"bessel_j2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_j2(x); })},
        {"bessel_y0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_y0(x); })},
        {"bessel_y1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_y1(x); })},
        {"bessel_y2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_bessel_y2(x); })},
        {"hermite_0", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(0, x); })},
        {"hermite_1", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(1, x); })},
        {"hermite_2", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(2, x); })},
        {"hermite_3", scalar_func_apply<double>([](double x) -> double { return gsl_sf_hermite(3, x); })},
        {"riemann_zeta", scalar_func_apply<double>([](double x) -> double { return gsl_sf_zeta(x); })},
    };

    // FIXME: check accuracy of this and this+test_func
    funs_cdx1 = {
        {"sin",
         scalar_func_apply<cdouble>([](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_sin_e); })},
        {"cos",
         scalar_func_apply<cdouble>([](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_cos_e); })},
        {"log",
         scalar_func_apply<cdouble>([](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_log_e); })},
        {"dilog", scalar_func_apply<cdouble>(
                      [](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_complex_dilog_e); })},
        {"lgamma", scalar_func_apply<cdouble>(
                       [](cdouble z) -> cdouble { return gsl_complex_wrapper(z, gsl_sf_lngamma_complex_e); })},
    };
}

std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1() {
    load_functions();
    return funs_dx1;
}

std::unordered_map<std::string, multi_eval_func<cdouble>> &get_funs_cdx1() {
    load_functions();
    return funs_cdx1;
}

} // namespace sf::functions::gsl
