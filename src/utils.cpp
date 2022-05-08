#include <sf_libraries.hpp>
#include <sf_utils.hpp>

namespace sf::utils {

host_info_t::host_info_t() {
    cpu_name = exec("grep -m1 'model name' /proc/cpuinfo | cut -d' ' --complement -f1-3");
    L1d = exec("lscpu | grep L1d | awk '{print $3}'");
    L1i = exec("lscpu | grep L1i | awk '{print $3}'");
    L2 = exec("lscpu | grep L2 | awk '{print $3}'");
    L3 = exec("lscpu | grep L3 | awk '{print $3}'");
}

toolchain_info_t::toolchain_info_t() {
#ifdef __GNUC__
    compiler = "gcc";
    compilervers =
        std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#endif

    libcvers = gnu_get_libc_version();
}

std::string exec(const char *cmd) {
    // https://stackoverflow.com/a/478960
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    result.pop_back();
    return result;
}

std::string get_alm_version() {
    std::string offset_str = "0x" + exec("objdump -t ../extern/amd-libm/lib/libalm.so --section=.rodata | grep -m1 "
                                         "ALM_VERSION_STRING | cut -d' ' -f 1");
    size_t offset = strtol(offset_str.c_str(), NULL, 0);
    FILE *obj = fopen("../extern/amd-libm/lib/libalm.so", "r");
    fseek(obj, offset, 0);
    char buf[16];
    fread(buf, sizeof(char), 16, obj);
    fclose(obj);
    return buf;
}

std::string get_sleef_version() {
    return std::to_string(SLEEF_VERSION_MAJOR) + "." + std::to_string(SLEEF_VERSION_MINOR) + "." +
           std::to_string(SLEEF_VERSION_PATCHLEVEL);
}

std::string get_af_version() {
    return std::to_string(VECTORCLASS_H / 10000) + "." + std::to_string((VECTORCLASS_H / 100) % 100) + "." +
           std::to_string(VECTORCLASS_H % 10);
}

std::string get_boost_version() {
    return std::to_string(BOOST_VERSION / 100000) + "." + std::to_string((BOOST_VERSION / 100) % 1000) + "." +
           std::to_string(BOOST_VERSION % 100);
}

std::string get_gsl_version() { return std::to_string(GSL_MAJOR_VERSION) + "." + std::to_string(GSL_MINOR_VERSION); }

std::string get_sctl_version() { return exec("cd ../extern/SCTL; git describe --tags"); }

std::string get_baobzi_version() { return exec("cd ../extern/baobzi; git describe --tags").substr(1); }

std::string get_eigen_version() {
    return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
           std::to_string(EIGEN_MINOR_VERSION);
}

} // namespace sf::utils
