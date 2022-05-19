#include <dlfcn.h>
#include <sf_libraries.hpp>

namespace sf::functions::amd {
std::unordered_map<std::string, multi_eval_func<float>> funs_fx1;
std::unordered_map<std::string, multi_eval_func<float>> funs_fx8;
std::unordered_map<std::string, multi_eval_func<double>> funs_dx1;
std::unordered_map<std::string, multi_eval_func<double>> funs_dx4;
using C_FUN1F = float (*)(float);
using C_FUN2F = float (*)(float, float);
using C_FUN1D = double (*)(double);
using C_FUN2D = double (*)(double, double);
using C_FX8_FUN1F = Vec8f (*)(Vec8f);
using C_FX8_FUN2F = Vec8f (*)(Vec8f, Vec8f);
using C_DX4_FUN1D = Vec4d (*)(Vec4d);
using C_DX4_FUN2D = Vec4d (*)(Vec4d, Vec4d);

void *handle = NULL;

C_FUN1F amd_sinf, amd_cosf, amd_tanf, amd_sinhf, amd_coshf, amd_tanhf, amd_asinf, amd_acosf, amd_atanf, amd_asinhf,
    amd_acoshf, amd_atanhf, amd_logf, amd_log2f, amd_log10f, amd_expf, amd_exp2f, amd_exp10f, amd_sqrtf;
C_FUN2F amd_powf;

C_FUN1D amd_sin, amd_cos, amd_tan, amd_sinh, amd_cosh, amd_tanh, amd_asin, amd_acos, amd_atan, amd_asinh, amd_acosh,
    amd_atanh, amd_log, amd_log2, amd_log10, amd_exp, amd_exp2, amd_exp10, amd_sqrt;
C_FUN2D amd_pow;

C_FX8_FUN1F amd_vrs8_sinf, amd_vrs8_cosf, amd_vrs8_tanf, amd_vrs8_logf, amd_vrs8_log2f, amd_vrs8_expf, amd_vrs8_exp2f;
C_FX8_FUN2F amd_vrs8_powf;

C_DX4_FUN1D amd_vrd4_sin, amd_vrd4_cos, amd_vrd4_tan, amd_vrd4_log, amd_vrd4_log2, amd_vrd4_exp, amd_vrd4_exp2;
C_DX4_FUN2D amd_vrd4_pow;

void load_functions() {
    if (handle)
        return;
    void *handle = dlopen("libalm.so", RTLD_NOW);

    amd_sinf = (C_FUN1F)dlsym(handle, "amd_sinf");
    amd_cosf = (C_FUN1F)dlsym(handle, "amd_cosf");
    amd_tanf = (C_FUN1F)dlsym(handle, "amd_tanf");
    amd_sinhf = (C_FUN1F)dlsym(handle, "amd_sinhf");
    amd_coshf = (C_FUN1F)dlsym(handle, "amd_coshf");
    amd_tanhf = (C_FUN1F)dlsym(handle, "amd_tanhf");
    amd_asinf = (C_FUN1F)dlsym(handle, "amd_asinf");
    amd_acosf = (C_FUN1F)dlsym(handle, "amd_acosf");
    amd_atanf = (C_FUN1F)dlsym(handle, "amd_atanf");
    amd_asinhf = (C_FUN1F)dlsym(handle, "amd_asinhf");
    amd_acoshf = (C_FUN1F)dlsym(handle, "amd_acoshf");
    amd_atanhf = (C_FUN1F)dlsym(handle, "amd_atanhf");
    amd_logf = (C_FUN1F)dlsym(handle, "amd_logf");
    amd_log2f = (C_FUN1F)dlsym(handle, "amd_log2f");
    amd_log10f = (C_FUN1F)dlsym(handle, "amd_log10f");
    amd_expf = (C_FUN1F)dlsym(handle, "amd_expf");
    amd_exp2f = (C_FUN1F)dlsym(handle, "amd_exp2f");
    amd_exp10f = (C_FUN1F)dlsym(handle, "amd_exp10f");
    amd_sqrtf = (C_FUN1F)dlsym(handle, "amd_sqrtf");
    amd_powf = (C_FUN2F)dlsym(handle, "amd_powf");

    amd_sin = (C_FUN1D)dlsym(handle, "amd_sin");
    amd_cos = (C_FUN1D)dlsym(handle, "amd_cos");
    amd_tan = (C_FUN1D)dlsym(handle, "amd_tan");
    amd_sinh = (C_FUN1D)dlsym(handle, "amd_sinh");
    amd_cosh = (C_FUN1D)dlsym(handle, "amd_cosh");
    amd_tanh = (C_FUN1D)dlsym(handle, "amd_tanh");
    amd_asin = (C_FUN1D)dlsym(handle, "amd_asin");
    amd_acos = (C_FUN1D)dlsym(handle, "amd_acos");
    amd_atan = (C_FUN1D)dlsym(handle, "amd_atan");
    amd_asinh = (C_FUN1D)dlsym(handle, "amd_asinh");
    amd_acosh = (C_FUN1D)dlsym(handle, "amd_acosh");
    amd_atanh = (C_FUN1D)dlsym(handle, "amd_atanh");
    amd_log = (C_FUN1D)dlsym(handle, "amd_log");
    amd_log2 = (C_FUN1D)dlsym(handle, "amd_log2");
    amd_log10 = (C_FUN1D)dlsym(handle, "amd_log10");
    amd_exp = (C_FUN1D)dlsym(handle, "amd_exp");
    amd_exp2 = (C_FUN1D)dlsym(handle, "amd_exp2");
    amd_exp10 = (C_FUN1D)dlsym(handle, "amd_exp10");
    amd_sqrt = (C_FUN1D)dlsym(handle, "amd_sqrt");
    amd_pow = (C_FUN2D)dlsym(handle, "amd_pow");

    amd_vrs8_sinf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_sinf");
    amd_vrs8_cosf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_cosf");
    amd_vrs8_tanf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_tanf");
    amd_vrs8_logf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_logf");
    amd_vrs8_log2f = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_log2f");
    amd_vrs8_expf = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_expf");
    amd_vrs8_exp2f = (C_FX8_FUN1F)dlsym(handle, "amd_vrs8_exp2f");
    amd_vrs8_powf = (C_FX8_FUN2F)dlsym(handle, "amd_vrs8_powf");

    amd_vrd4_sin = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_sin");
    amd_vrd4_cos = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_cos");
    amd_vrd4_tan = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_tan");
    amd_vrd4_log = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_log");
    amd_vrd4_log2 = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_log2");
    amd_vrd4_exp = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_exp");
    amd_vrd4_exp2 = (C_DX4_FUN1D)dlsym(handle, "amd_vrd4_exp2");
    amd_vrd4_pow = (C_DX4_FUN2D)dlsym(handle, "amd_vrd4_pow");

    funs_fx1 = {
        {"sin", scalar_func_apply<float>([](float x) -> float { return amd_sinf(x); })},
        {"cos", scalar_func_apply<float>([](float x) -> float { return amd_cosf(x); })},
        {"tan", scalar_func_apply<float>([](float x) -> float { return amd_tanf(x); })},
        {"sinh", scalar_func_apply<float>([](float x) -> float { return amd_sinhf(x); })},
        {"cosh", scalar_func_apply<float>([](float x) -> float { return amd_coshf(x); })},
        {"tanh", scalar_func_apply<float>([](float x) -> float { return amd_tanhf(x); })},
        {"asin", scalar_func_apply<float>([](float x) -> float { return amd_asinf(x); })},
        {"acos", scalar_func_apply<float>([](float x) -> float { return amd_acosf(x); })},
        {"atan", scalar_func_apply<float>([](float x) -> float { return amd_atanf(x); })},
        {"asinh", scalar_func_apply<float>([](float x) -> float { return amd_asinhf(x); })},
        {"acosh", scalar_func_apply<float>([](float x) -> float { return amd_acoshf(x); })},
        {"atanh", scalar_func_apply<float>([](float x) -> float { return amd_atanhf(x); })},
        {"log", scalar_func_apply<float>([](float x) -> float { return amd_logf(x); })},
        {"log2", scalar_func_apply<float>([](float x) -> float { return amd_log2f(x); })},
        {"log10", scalar_func_apply<float>([](float x) -> float { return amd_log10f(x); })},
        {"exp", scalar_func_apply<float>([](float x) -> float { return amd_expf(x); })},
        {"exp2", scalar_func_apply<float>([](float x) -> float { return amd_exp2f(x); })},
        {"exp10", scalar_func_apply<float>([](float x) -> float { return amd_exp10f(x); })},
        {"sqrt", scalar_func_apply<float>([](float x) -> float { return amd_sqrtf(x); })},
        {"pow3.5", scalar_func_apply<float>([](float x) -> float { return amd_powf(x, 3.5); })},
        {"pow13", scalar_func_apply<float>([](float x) -> float { return amd_powf(x, 13); })},
    };

    funs_dx1 = {
        {"sin", scalar_func_apply<double>([](double x) -> double { return amd_sin(x); })},
        {"cos", scalar_func_apply<double>([](double x) -> double { return amd_cos(x); })},
        {"tan", scalar_func_apply<double>([](double x) -> double { return amd_tan(x); })},
        {"sinh", scalar_func_apply<double>([](double x) -> double { return amd_sinh(x); })},
        {"cosh", scalar_func_apply<double>([](double x) -> double { return amd_cosh(x); })},
        {"tanh", scalar_func_apply<double>([](double x) -> double { return amd_tanh(x); })},
        {"asin", scalar_func_apply<double>([](double x) -> double { return amd_asin(x); })},
        {"acos", scalar_func_apply<double>([](double x) -> double { return amd_acos(x); })},
        {"atan", scalar_func_apply<double>([](double x) -> double { return amd_atan(x); })},
        {"asinh", scalar_func_apply<double>([](double x) -> double { return amd_asinh(x); })},
        {"acosh", scalar_func_apply<double>([](double x) -> double { return amd_acosh(x); })},
        {"atanh", scalar_func_apply<double>([](double x) -> double { return amd_atanh(x); })},
        {"log", scalar_func_apply<double>([](double x) -> double { return amd_log(x); })},
        {"log2", scalar_func_apply<double>([](double x) -> double { return amd_log2(x); })},
        {"log10", scalar_func_apply<double>([](double x) -> double { return amd_log10(x); })},
        {"exp", scalar_func_apply<double>([](double x) -> double { return amd_exp(x); })},
        {"exp2", scalar_func_apply<double>([](double x) -> double { return amd_exp2(x); })},
        {"exp10", scalar_func_apply<double>([](double x) -> double { return amd_exp10(x); })},
        {"sqrt", scalar_func_apply<double>([](double x) -> double { return amd_sqrt(x); })},
        {"pow3.5", scalar_func_apply<double>([](double x) -> double { return amd_pow(x, 3.5); })},
        {"pow13", scalar_func_apply<double>([](double x) -> double { return amd_pow(x, 13); })},
    };

    funs_dx4 = {
        {"sin", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_sin(x); })},
        {"cos", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_cos(x); })},
        {"tan", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_tan(x); })},
        {"log", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_log(x); })},
        {"log2", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_log2(x); })},
        {"exp", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_exp(x); })},
        {"exp2", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_exp2(x); })},
        {"pow3.5", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_pow(x, Vec4d{3.5}); })},
        {"pow13", vec_func_map<Vec4d, double>([](Vec4d x) -> Vec4d { return amd_vrd4_pow(x, Vec4d{13}); })},
    };

    funs_fx8 = {
        {"sin", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_sinf(x); })},
        {"cos", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_cosf(x); })},
        {"tan", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_tanf(x); })},
        {"log", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_logf(x); })},
        {"log2", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_log2f(x); })},
        {"exp", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_expf(x); })},
        {"exp2", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_exp2f(x); })},
        {"pow3.5", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_powf(x, Vec8f{3.5}); })},
        {"pow13", vec_func_map<Vec8f, float>([](Vec8f x) -> Vec8f { return amd_vrs8_powf(x, Vec8f{13}); })},
    };
}

std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx1() {
    load_functions();
    return funs_fx1;
}
std::unordered_map<std::string, multi_eval_func<float>> &get_funs_fx8() {
    load_functions();
    return funs_fx8;
}
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx1() {
    load_functions();
    return funs_dx1;
}
std::unordered_map<std::string, multi_eval_func<double>> &get_funs_dx4() {
    load_functions();
    return funs_dx4;
}
} // namespace sf::functions::amd
