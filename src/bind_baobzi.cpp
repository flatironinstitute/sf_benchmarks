#include <dlfcn.h>
#include <string>
#include <unordered_map>

#include <sf_libraries.hpp>
#include <baobzi.hpp>

namespace sf::functions::baobzi {
using ::baobzi::Baobzi;

double baobzi_fun_wrapper(const double *x, const void *data) {
    auto *myfun = (std::function<double(double)> *)data;
    return (*myfun)(*x);
}

std::shared_ptr<Baobzi> create_baobzi_func(void *infun, const std::pair<double, double> &domain) {
    baobzi_input_t input = {.func = baobzi_fun_wrapper,
                            .data = infun,
                            .dim = 1,
                            .order = 8,
                            .tol = 1E-10,
                            .minimum_leaf_fraction = 0.6,
                            .split_multi_eval = 0};
    double hl = 0.5 * (domain.second - domain.first);
    double center = domain.first + hl;

    return std::shared_ptr<Baobzi>(new Baobzi(&input, &center, &hl));
}

std::unordered_map<std::string, std::shared_ptr<::baobzi::Baobzi>> baobzi_funs;
std::unordered_map<std::string, std::function<double(double)>> potential_baobzi_funs{
    {"bessel_Y0", [](double x) -> double { return gsl_sf_bessel_Y0(x); }},
    {"bessel_Y1", [](double x) -> double { return gsl_sf_bessel_Y1(x); }},
    {"bessel_Y2", [](double x) -> double { return gsl_sf_bessel_Yn(2, x); }},
    {"bessel_I0", [](double x) -> double { return gsl_sf_bessel_I0(x); }},
    {"bessel_I1", [](double x) -> double { return gsl_sf_bessel_I1(x); }},
    {"bessel_I2", [](double x) -> double { return gsl_sf_bessel_In(2, x); }},
    {"bessel_J0", [](double x) -> double { return gsl_sf_bessel_J0(x); }},
    {"bessel_J1", [](double x) -> double { return gsl_sf_bessel_J1(x); }},
    {"bessel_J2", [](double x) -> double { return gsl_sf_bessel_Jn(2, x); }},
    {"hermite_0", [](double x) -> double { return gsl_sf_hermite(0, x); }},
    {"hermite_1", [](double x) -> double { return gsl_sf_hermite(1, x); }},
    {"hermite_2", [](double x) -> double { return gsl_sf_hermite(2, x); }},
    {"hermite_3", [](double x) -> double { return gsl_sf_hermite(3, x); }},
};

std::unordered_map<std::string, std::shared_ptr<::baobzi::Baobzi>> &
get_funs_dx1(std::set<std::string> &keys_to_eval, std::unordered_map<std::string, configuration_t> &params) {
    for (auto &key : keys_to_eval) {
        if (potential_baobzi_funs.count(key) && !baobzi_funs.count(key)) {
            std::cerr << "Creating baobzi function '" + key + "'.\n";
            auto &param = params[key];
            std::pair domain = std::make_pair(param.lbound, param.ubound);
            baobzi_funs[key] = create_baobzi_func((void *)(&potential_baobzi_funs.at(key)), domain);
        }
    }

    return baobzi_funs;
}

} // namespace sf::functions::misc
