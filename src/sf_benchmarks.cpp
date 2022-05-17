#include <sf_benchmarks.hpp>
#include <sf_utils.hpp>

#include <gnu/libc-version.h>
#include <unistd.h>

host_info_t::host_info_t() {
    cpuname = sf::utils::exec("grep -m1 'model name' /proc/cpuinfo | cut -d' ' --complement -f1-3");
    L1d = std::to_string(sysconf(_SC_LEVEL1_DCACHE_SIZE));
    L1i = std::to_string(sysconf(_SC_LEVEL1_ICACHE_SIZE));
    L2 = std::to_string(sysconf(_SC_LEVEL2_CACHE_SIZE));
    L3 = std::to_string(sysconf(_SC_LEVEL3_CACHE_SIZE));
}

toolchain_info_t::toolchain_info_t() {
#if defined(__clang__)
    compiler = "clang";
    compilervers = std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
                   std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    compiler = "gcc";
    compilervers =
        std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#endif

    libcvers = gnu_get_libc_version();
}
