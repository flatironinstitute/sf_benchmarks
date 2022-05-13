#include <sf_libraries.hpp>

namespace sf::functions::eigen {
std::unordered_map<std::string, OPS> funs = {
    {"sin", OPS::sin},         {"cos", OPS::cos},      {"tan", OPS::tan},     {"sinh", OPS::sinh},
    {"cosh", OPS::cosh},       {"tanh", OPS::tanh},    {"exp", OPS::exp},     {"log", OPS::log},
    {"log10", OPS::log10},     {"pow3.5", OPS::pow35}, {"pow13", OPS::pow13}, {"asin", OPS::asin},
    {"acos", OPS::acos},       {"atan", OPS::atan},    {"asinh", OPS::asinh}, {"atanh", OPS::atanh},
    {"acosh", OPS::acosh},     {"erf", OPS::erf},      {"erfc", OPS::erfc},   {"lgamma", OPS::lgamma},
    {"digamma", OPS::digamma}, {"ndtri", OPS::ndtri},  {"sqrt", OPS::sqrt},   {"rsqrt", OPS::rsqrt},
};

std::unordered_map<std::string, OPS> &get_funs() { return funs; }
} // namespace sf::functions::eigen
