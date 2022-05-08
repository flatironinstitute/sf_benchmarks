#include <dlfcn.h>
#include <string>
#include <unordered_map>

#include <sf_libraries.hpp>

namespace sf::functions::stl {
std::unordered_map<std::string, multi_eval_func<float>> funs_fx1 = {
    {"tgamma", scalar_func_apply<float>([](float x) -> float { return std::tgamma(x); })},
    {"lgamma", scalar_func_apply<float>([](float x) -> float { return std::lgamma(x); })},
    {"sin", scalar_func_apply<float>([](float x) -> float { return std::sin(x); })},
    {"cos", scalar_func_apply<float>([](float x) -> float { return std::cos(x); })},
    {"tan", scalar_func_apply<float>([](float x) -> float { return std::tan(x); })},
    {"asin", scalar_func_apply<float>([](float x) -> float { return std::asin(x); })},
    {"acos", scalar_func_apply<float>([](float x) -> float { return std::acos(x); })},
    {"atan", scalar_func_apply<float>([](float x) -> float { return std::atan(x); })},
    {"asin", scalar_func_apply<float>([](float x) -> float { return std::asin(x); })},
    {"acos", scalar_func_apply<float>([](float x) -> float { return std::acos(x); })},
    {"atan", scalar_func_apply<float>([](float x) -> float { return std::atan(x); })},
    {"sinh", scalar_func_apply<float>([](float x) -> float { return std::sinh(x); })},
    {"cosh", scalar_func_apply<float>([](float x) -> float { return std::cosh(x); })},
    {"tanh", scalar_func_apply<float>([](float x) -> float { return std::tanh(x); })},
    {"asinh", scalar_func_apply<float>([](float x) -> float { return std::asinh(x); })},
    {"acosh", scalar_func_apply<float>([](float x) -> float { return std::acosh(x); })},
    {"atanh", scalar_func_apply<float>([](float x) -> float { return std::atanh(x); })},
    {"sin_pi", scalar_func_apply<float>([](float x) -> float { return std::sin(M_PI * x); })},
    {"cos_pi", scalar_func_apply<float>([](float x) -> float { return std::cos(M_PI * x); })},
    {"erf", scalar_func_apply<float>([](float x) -> float { return std::erf(x); })},
    {"erfc", scalar_func_apply<float>([](float x) -> float { return std::erfc(x); })},
    {"log", scalar_func_apply<float>([](float x) -> float { return std::log(x); })},
    {"log2", scalar_func_apply<float>([](float x) -> float { return std::log2(x); })},
    {"log10", scalar_func_apply<float>([](float x) -> float { return std::log10(x); })},
    {"exp", scalar_func_apply<float>([](float x) -> float { return std::exp(x); })},
    {"exp2", scalar_func_apply<float>([](float x) -> float { return std::exp2(x); })},
    {"exp10", scalar_func_apply<float>([](float x) -> float { return exp10(x); })},
    {"sqrt", scalar_func_apply<float>([](float x) -> float { return std::sqrt(x); })},
    {"rsqrt", scalar_func_apply<float>([](float x) -> float { return 1.0 / std::sqrt(x); })},
    {"pow3.5", scalar_func_apply<float>([](float x) -> float { return std::pow(x, 3.5); })},
    {"pow13", scalar_func_apply<float>([](float x) -> float { return std::pow(x, 13); })},
};

std::unordered_map<std::string, multi_eval_func<double>> funs_dx1 = {
    {"tgamma", scalar_func_apply<double>([](double x) -> double { return std::tgamma(x); })},
    {"lgamma", scalar_func_apply<double>([](double x) -> double { return std::lgamma(x); })},
    {"sin", scalar_func_apply<double>([](double x) -> double { return std::sin(x); })},
    {"cos", scalar_func_apply<double>([](double x) -> double { return std::cos(x); })},
    {"tan", scalar_func_apply<double>([](double x) -> double { return std::tan(x); })},
    {"asin", scalar_func_apply<double>([](double x) -> double { return std::asin(x); })},
    {"acos", scalar_func_apply<double>([](double x) -> double { return std::acos(x); })},
    {"atan", scalar_func_apply<double>([](double x) -> double { return std::atan(x); })},
    {"asin", scalar_func_apply<double>([](double x) -> double { return std::asin(x); })},
    {"acos", scalar_func_apply<double>([](double x) -> double { return std::acos(x); })},
    {"atan", scalar_func_apply<double>([](double x) -> double { return std::atan(x); })},
    {"sinh", scalar_func_apply<double>([](double x) -> double { return std::sinh(x); })},
    {"cosh", scalar_func_apply<double>([](double x) -> double { return std::cosh(x); })},
    {"tanh", scalar_func_apply<double>([](double x) -> double { return std::tanh(x); })},
    {"asinh", scalar_func_apply<double>([](double x) -> double { return std::asinh(x); })},
    {"acosh", scalar_func_apply<double>([](double x) -> double { return std::acosh(x); })},
    {"atanh", scalar_func_apply<double>([](double x) -> double { return std::atanh(x); })},
    {"sin_pi", scalar_func_apply<double>([](double x) -> double { return std::sin(M_PI * x); })},
    {"cos_pi", scalar_func_apply<double>([](double x) -> double { return std::cos(M_PI * x); })},
    {"erf", scalar_func_apply<double>([](double x) -> double { return std::erf(x); })},
    {"erfc", scalar_func_apply<double>([](double x) -> double { return std::erfc(x); })},
    {"log", scalar_func_apply<double>([](double x) -> double { return std::log(x); })},
    {"log2", scalar_func_apply<double>([](double x) -> double { return std::log2(x); })},
    {"log10", scalar_func_apply<double>([](double x) -> double { return std::log10(x); })},
    {"exp", scalar_func_apply<double>([](double x) -> double { return std::exp(x); })},
    {"exp2", scalar_func_apply<double>([](double x) -> double { return std::exp2(x); })},
    {"exp10", scalar_func_apply<double>([](double x) -> double { return exp10(x); })},
    {"sqrt", scalar_func_apply<double>([](double x) -> double { return std::sqrt(x); })},
    {"rsqrt", scalar_func_apply<double>([](double x) -> double { return 1.0 / std::sqrt(x); })},
    {"pow3.5", scalar_func_apply<double>([](double x) -> double { return std::pow(x, 3.5); })},
    {"pow13", scalar_func_apply<double>([](double x) -> double { return std::pow(x, 13); })},
};

std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1() { return funs_fx1; }
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1() { return funs_dx1; }
} // namespace sf::functions::stl