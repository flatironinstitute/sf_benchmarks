#include <sf_benchmarks.hpp>
#include <sf_utils.hpp>

#include <gnu/libc-version.h>

host_info_t::host_info_t() {
    cpuname = sf::utils::exec("grep -m1 'model name' /proc/cpuinfo | cut -d' ' --complement -f1-3");
    L1d = sf::utils::exec("lscpu | grep L1d | awk '{print $3}'");
    L1i = sf::utils::exec("lscpu | grep L1i | awk '{print $3}'");
    L2 = sf::utils::exec("lscpu | grep L2 | awk '{print $3}'");
    L3 = sf::utils::exec("lscpu | grep L3 | awk '{print $3}'");
}

toolchain_info_t::toolchain_info_t() {
#if defined(__clang__)
    compiler = "clang";
    compilervers =
        std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    compiler = "gcc";
    compilervers =
        std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#endif

    libcvers = gnu_get_libc_version();
}
