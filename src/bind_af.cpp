#include <sf_libraries.hpp>

namespace sf::functions::af {
std::unordered_map<std::string, multi_eval_func<float>> funs_fx8 = {
    {"sqrt", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return sqrt(x); })},
    {"sin", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return sin(x); })},
    {"cos", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return cos(x); })},
    {"tan", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return tan(x); })},
    {"sinh", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return sinh(x); })},
    {"cosh", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return cosh(x); })},
    {"tanh", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return tanh(x); })},
    {"asinh", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return asinh(x); })},
    {"acosh", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return acosh(x); })},
    {"atanh", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return atanh(x); })},
    {"asin", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return asin(x); })},
    {"acos", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return acos(x); })},
    {"atan", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return atan(x); })},
    {"exp", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return exp(x); })},
    {"exp2", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return exp2(x); })},
    {"exp10", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return exp10(x); })},
    {"log", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return log(x); })},
    {"log2", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return log2(x); })},
    {"log10", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return log10(x); })},
    {"pow3.5", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return pow(x, 3.5); })},
    {"pow13", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return pow_const(x, 13); })},
};

std::unordered_map<std::string, multi_eval_func<double>> funs_dx4 = {
    {"sqrt", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return sqrt(x); })},
    {"sin", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return sin(x); })},
    {"cos", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return cos(x); })},
    {"tan", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return tan(x); })},
    {"sinh", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return sinh(x); })},
    {"cosh", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return cosh(x); })},
    {"tanh", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return tanh(x); })},
    {"asinh", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return asinh(x); })},
    {"acosh", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return acosh(x); })},
    {"atanh", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return atanh(x); })},
    {"asin", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return asin(x); })},
    {"acos", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return acos(x); })},
    {"atan", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return atan(x); })},
    {"exp", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return exp(x); })},
    {"exp2", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return exp2(x); })},
    {"exp10", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return exp10(x); })},
    {"log", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return log(x); })},
    {"log2", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return log2(x); })},
    {"log10", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return log10(x); })},
    {"pow3.5", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return pow(x, 3.5); })},
    {"pow13", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return pow_const(x, 13); })},
};

#ifdef __AVX512F__
std::unordered_map<std::string, multi_eval_func<float>> funs_fx16 = {
    {"memcpy", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return x; })},
    {"memset", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return Vec16f{0.0}; })},
    {"sqrt", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return sqrt(x); })},
    {"sin", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return sin(x); })},
    {"cos", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return cos(x); })},
    {"tan", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return tan(x); })},
    {"sinh", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return sinh(x); })},
    {"cosh", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return cosh(x); })},
    {"tanh", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return tanh(x); })},
    {"asinh", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return asinh(x); })},
    {"acosh", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return acosh(x); })},
    {"atanh", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return atanh(x); })},
    {"asin", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return asin(x); })},
    {"acos", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return acos(x); })},
    {"atan", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return atan(x); })},
    {"exp", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return exp(x); })},
    {"exp2", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return exp2(x); })},
    {"exp10", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return exp10(x); })},
    {"log", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return log(x); })},
    {"log2", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return log2(x); })},
    {"log10", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return log10(x); })},
    {"pow3.5", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return pow(x, 3.5); })},
    {"pow13", vec_func_map<Vec16f, float>([](Vec16f x) -> Vec16f { return pow_const(x, 13); })},
};

std::unordered_map<std::string, multi_eval_func<double>> funs_dx8 = {
    {"memset", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return Vec8d{0.0}; })},
    {"memcpy", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return x; })},
    {"sqrt", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return sqrt(x); })},
    {"sin", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return sin(x); })},
    {"cos", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return cos(x); })},
    {"tan", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return tan(x); })},
    {"sinh", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return sinh(x); })},
    {"cosh", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return cosh(x); })},
    {"tanh", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return tanh(x); })},
    {"asinh", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return asinh(x); })},
    {"acosh", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return acosh(x); })},
    {"atanh", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return atanh(x); })},
    {"asin", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return asin(x); })},
    {"acos", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return acos(x); })},
    {"atan", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return atan(x); })},
    {"exp", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return exp(x); })},
    {"exp2", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return exp2(x); })},
    {"exp10", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return exp10(x); })},
    {"log", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return log(x); })},
    {"log2", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return log2(x); })},
    {"log10", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return log10(x); })},
    {"pow3.5", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return pow(x, 3.5); })},
    {"pow13", vec_func_map<Vec8d, double>([](Vec8d x) -> Vec8d { return pow_const(x, 13); })},
};
#else
std::unordered_map<std::string, multi_eval_func<float>> funs_fx16;
std::unordered_map<std::string, multi_eval_func<double>> funs_dx8;
#endif

std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8() { return funs_fx8; }
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4() { return funs_dx4; }
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx16() { return funs_fx16; }
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx8() { return funs_dx8; }

} // namespace sf::functions::af
