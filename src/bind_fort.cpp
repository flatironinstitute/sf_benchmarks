#include <sf_libraries.hpp>

namespace sf::functions::fort {
std::unordered_map<std::string, multi_eval_func<double>> funs_dx1 = {
    {"bessel_Y0", scalar_func_apply<double>([](double x) -> double {
         int n = 0;
         double y;
         fort_bessel_yn_(&n, &x, &y);
         return y;
     })},
    {"bessel_J0", scalar_func_apply<double>([](double x) -> double {
         int n = 0;
         double y;
         fort_bessel_jn_(&n, &x, &y);
         return y;
     })},
};

std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1() { return funs_dx1; }
} // namespace sf::functions::fort
