#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include <sf_benchmarks.hpp>
#include <sf_libraries.hpp>
#include <sf_utils.hpp>

#include <sqlite_orm/sqlite_orm.h>

sf::utils::toolchain_info_t toolchain_info;
sf::utils::host_info_t host_info;
std::unordered_map<std::string, sf::utils::library_info_t> libraries_info = {
    {"agnerfog", {0, "agnerfog", sf::utils::get_af_version()}},
    {"amdlibm", {0, "amdlibm", sf::utils::get_alm_version()}},
    {"baobzi", {0, "baobzi", sf::utils::get_baobzi_version()}},
    {"boost", {0, "boost", sf::utils::get_boost_version()}},
    {"eigen", {0, "eigen", sf::utils::get_eigen_version()}},
    {"gsl", {0, "gsl", sf::utils::get_gsl_version()}},
    {"fort", {0, "fort", "NA"}},
    {"misc", {0, "misc", "NA"}},
    {"sctl", {0, "sctl", sf::utils::get_sctl_version()}},
    {"sleef", {0, "sleef", sf::utils::get_sleef_version()}},
    {"stl", {0, "stl", "NA"}},
};

struct run_t {
    int id;
    std::string time;
    std::unique_ptr<int> host;
    std::unique_ptr<int> toolchain;
};

run_t current_run;

struct measurement_t {
    int id;
    std::unique_ptr<int> run;
    std::unique_ptr<int> library;
    std::unique_ptr<int> configuration;
    sf::utils::library_info_t library_copy;
    configuration_t config_copy;
    int nelem = 0;
    int nrepeat = 0;
    int veclev = 0;
    double megaevalspersec = 0;
    double meanevaltime = 0;
    double stddev = 0;
    double maxerr = 0;

    explicit operator bool() const { return nrepeat; }
    friend std::ostream &operator<<(std::ostream &, const measurement_t &);
};

std::ostream &operator<<(std::ostream &os, const measurement_t &meas) {

    using std::left;
    using std::setw;

    if (meas) {
        std::string label = meas.config_copy.func + "_" + meas.library_copy.name + "_" + meas.config_copy.ftype + "x" +
                            std::to_string(meas.veclev);

        os.precision(6);
        os << left << setw(25) << label + ": " << left << setw(15) << meas.megaevalspersec;
        os.precision(15);
        os << left << setw(15) << meas.meanevaltime << left << setw(5) << " ";
        os.precision(5);
        os << "[" << meas.config_copy.lbound << ", " << meas.config_copy.ubound << "]" << std::endl;
    }
    return os;
}

#define EIGEN_CASE(OP)                                                                                                 \
    case sf::functions::eigen::OPS::OP: {                                                                              \
        res = x.array().OP();                                                                                          \
        break;                                                                                                         \
    }

template <typename FUN_T, typename VAL_T>
measurement_t test_func(const FUN_T &f, int veclev, sf::utils::library_info_t &library_info, configuration_t &config,
                        const Eigen::VectorX<VAL_T> &x_in, int n_repeat) {
    if (!f)
        return measurement_t();
    const std::string label = library_info.name + "_" + config.func;

    Eigen::VectorX<VAL_T> x = sf::utils::transform_domain(x_in, config.lbound, config.ubound);

    size_t res_size = x.size();
    size_t n_evals = x.size() * n_repeat;
    if constexpr (std::is_same_v<FUN_T, fun_cdx1_x2>)
        res_size *= 2;

    Eigen::VectorX<VAL_T> res(res_size);
    VAL_T *resptr = res.data();

    sf::utils::timer timer;
    for (long k = 0; k < n_repeat; k++) {
        if constexpr (std::is_same_v<FUN_T, fun_cdx1_x2>) {
            for (std::size_t i = 0; i < x.size(); ++i) {
                std::tie(resptr[i * 2], resptr[i * 2 + 1]) = f(x[i]);
            }
        } else if constexpr (std::is_same_v<FUN_T, std::shared_ptr<baobzi::Baobzi>>) {
            (*f)(x.data(), resptr, x.size());
        } else if constexpr (std::is_same_v<FUN_T, sf::functions::eigen::OPS>) {
            switch (f) {
                EIGEN_CASE(cos)
                EIGEN_CASE(sin)
                EIGEN_CASE(tan)
                EIGEN_CASE(cosh)
                EIGEN_CASE(sinh)
                EIGEN_CASE(tanh)
                EIGEN_CASE(exp)
                EIGEN_CASE(log)
                EIGEN_CASE(log10)
                EIGEN_CASE(asin)
                EIGEN_CASE(acos)
                EIGEN_CASE(atan)
                EIGEN_CASE(asinh)
                EIGEN_CASE(acosh)
                EIGEN_CASE(atanh)
                EIGEN_CASE(erf)
                EIGEN_CASE(erfc)
                EIGEN_CASE(lgamma)
                EIGEN_CASE(digamma)
                EIGEN_CASE(ndtri)
                EIGEN_CASE(sqrt)
                EIGEN_CASE(rsqrt)
            case sf::functions::eigen::OPS::pow35: {
                res = x.array().pow(3.5);
                break;
            }
            case sf::functions::eigen::OPS::pow13: {
                res = x.array().pow(13);
                break;
            }
            }
        } else {
            f(x.data(), resptr, x.size());
        }
    }
    timer.stop();

    measurement_t meas;
    meas.config_copy = config;
    meas.library_copy = library_info;

    meas.run = std::make_unique<int>(current_run.id);
    meas.configuration = std::make_unique<int>(config.id);
    meas.library = std::make_unique<int>(library_info.id);
    meas.nelem = x.size();
    meas.nrepeat = n_repeat;
    meas.megaevalspersec = n_evals / timer.elapsed() / 1E6;
    meas.meanevaltime = timer.elapsed() / n_evals / 1E-9;
    meas.veclev = veclev;

    return meas;
}
#undef EIGEN_CASE

std::set<std::string> parse_args(int argc, char *argv[]) {
    // lol: "parse"
    std::set<std::string> res;
    for (int i = 0; i < argc; ++i)
        res.insert(argv[i]);

    return res;
}

inline auto init_storage(const std::string &path) {
    using namespace sqlite_orm;
    using sf::utils::host_info_t;
    using sf::utils::library_info_t;
    using sf::utils::toolchain_info_t;

    auto storage = make_storage(
        "db.sqlite",
        make_table(
            "hosts", make_column("id", &host_info_t::id, autoincrement(), primary_key()),
            make_column("cpuname", &host_info_t::cpuname, unique()), make_column("cpuclock", &host_info_t::cpuclock),
            make_column("cpuclockmax", &host_info_t::cpuclockmax), make_column("memclock", &host_info_t::memclock),
            make_column("l1dcache", &host_info_t::L1d), make_column("l1icache", &host_info_t::L1i),
            make_column("l2cache", &host_info_t::L2), make_column("l3cache", &host_info_t::L3)),
        make_table("configurations", make_column("id", &configuration_t::id, autoincrement(), primary_key()),
                   make_column("func", &configuration_t::func), make_column("ftype", &configuration_t::ftype),
                   make_column("lbound", &configuration_t::lbound), make_column("ubound", &configuration_t::ubound),
                   make_column("ilbound", &configuration_t::ilbound), make_column("iubound", &configuration_t::iubound),
                   sqlite_orm::unique(&configuration_t::func, &configuration_t::ftype, &configuration_t::lbound,
                                      &configuration_t::ubound, &configuration_t::ilbound, &configuration_t::iubound)),
        make_table("toolchains", make_column("id", &toolchain_info_t::id, autoincrement(), primary_key()),
                   make_column("compiler", &toolchain_info_t::compiler),
                   make_column("compilervers", &toolchain_info_t::compilervers),
                   make_column("libcvers", &toolchain_info_t::libcvers),
                   sqlite_orm::unique(&toolchain_info_t::compiler, &toolchain_info_t::compilervers,
                                      &toolchain_info_t::libcvers)),
        make_table("libraries", make_column("id", &library_info_t::id, autoincrement(), primary_key()),
                   make_column("name", &library_info_t::name), make_column("version", &library_info_t::version),
                   sqlite_orm::unique(&library_info_t::name, &library_info_t::version)),
        make_table("runs", make_column("id", &run_t::id, autoincrement(), primary_key()),
                   make_column("time", &run_t::time), make_column("host", &run_t::host),
                   make_column("toolchain", &run_t::toolchain), foreign_key(&run_t::host).references(&host_info_t::id),
                   foreign_key(&run_t::toolchain).references(&toolchain_info_t::id)),
        make_table(
            "measurements", make_column("id", &measurement_t::id, autoincrement(), primary_key()),
            make_column("run", &measurement_t::run), make_column("library", &measurement_t::library),
            make_column("configuration", &measurement_t::configuration), make_column("nelem", &measurement_t::nelem),
            make_column("nrepeat", &measurement_t::nrepeat), make_column("veclev", &measurement_t::veclev),
            make_column("megaevalspersec", &measurement_t::megaevalspersec),
            make_column("meanevaltime", &measurement_t::meanevaltime), make_column("stddev", &measurement_t::stddev),
            make_column("maxerr", &measurement_t::maxerr), foreign_key(&measurement_t::run).references(&run_t::id),
            foreign_key(&measurement_t::library).references(&library_info_t::id),
            foreign_key(&measurement_t::configuration).references(&configuration_t::id)));

    storage.sync_schema();
    auto host_ids =
        storage.select(columns(&host_info_t::id), where(is_equal(&host_info_t::cpuname, host_info.cpuname)));
    if (host_ids.size() == 0)
        host_info.id = storage.insert(host_info);
    else
        host_info.id = std::get<int>(host_ids[0]);

    auto toolchain_ids = storage.select(columns(&toolchain_info_t::id),
                                        where(is_equal(&toolchain_info_t::compiler, toolchain_info.compiler) and
                                              is_equal(&toolchain_info_t::compilervers, toolchain_info.compilervers) and
                                              is_equal(&toolchain_info_t::libcvers, toolchain_info.libcvers)));
    if (toolchain_ids.size() == 0)
        toolchain_info.id = storage.insert(toolchain_info);
    else
        toolchain_info.id = std::get<int>(toolchain_ids[0]);

    for (auto &[name, lib] : libraries_info) {
        auto library_ids =
            storage.select(columns(&library_info_t::id), where(is_equal(&library_info_t::name, lib.name) and
                                                               is_equal(&library_info_t::version, lib.version)));
        if (library_ids.size() == 0)
            lib.id = storage.insert(lib);
        else
            lib.id = std::get<int>(library_ids[0]);
    }

    current_run.time = storage.select(datetime("now")).front();
    current_run.toolchain = std::make_unique<int>(toolchain_info.id);
    current_run.host = std::make_unique<int>(host_info.id);
    current_run.id = storage.insert(current_run);

    return storage;
}

using Storage = decltype(init_storage(""));

int main(int argc, char *argv[]) {
    Storage storage = init_storage("db.sqlite");

    std::cout << host_info.cpuname << std::endl;
    std::cout << "    " + toolchain_info.compiler + ": " + toolchain_info.compilervers << std::endl;
    std::cout << "    libc: " + toolchain_info.libcvers << std::endl;
    for (auto &[key, lib] : libraries_info)
        std::cout << "    " + lib.name + ": " + lib.version << std::endl;

    std::set<std::string> input_keys = parse_args(argc - 1, argv + 1);

    auto &af_funs_dx4 = sf::functions::af::get_funs_dx4();
    auto &af_funs_dx8 = sf::functions::af::get_funs_dx8();
    auto &af_funs_fx8 = sf::functions::af::get_funs_fx8();
    auto &af_funs_fx16 = sf::functions::af::get_funs_fx16();

    auto &amdlibm_funs_dx1 = sf::functions::amd::get_funs_dx1();
    auto &amdlibm_funs_dx4 = sf::functions::amd::get_funs_dx4();
    auto &amdlibm_funs_fx1 = sf::functions::amd::get_funs_fx1();
    auto &amdlibm_funs_fx8 = sf::functions::amd::get_funs_fx8();

    auto &boost_funs_fx1 = sf::functions::boost::get_funs_fx1();
    auto &boost_funs_dx1 = sf::functions::boost::get_funs_dx1();

    auto &eigen_funs = sf::functions::eigen::get_funs();

    auto &fort_funs = sf::functions::fort::get_funs_dx1();

    auto &gsl_funs = sf::functions::gsl::get_funs_dx1();
    auto &gsl_complex_funs = sf::functions::gsl::get_funs_cdx1();

    auto &misc_funs_cdx1_x2 = sf::functions::misc::get_funs_cdx1_x2();

    auto &sctl_funs_dx4 = sf::functions::SCTL::get_funs_dx4();
    auto &sctl_funs_dx8 = sf::functions::SCTL::get_funs_dx8();
    auto &sctl_funs_fx8 = sf::functions::SCTL::get_funs_fx8();
    auto &sctl_funs_fx16 = sf::functions::SCTL::get_funs_fx16();

    auto &sleef_funs_dx1 = sf::functions::sleef::get_funs_dx1();
    auto &sleef_funs_dx4 = sf::functions::sleef::get_funs_dx4();
    auto &sleef_funs_dx8 = sf::functions::sleef::get_funs_dx8();
    auto &sleef_funs_fx1 = sf::functions::sleef::get_funs_fx1();
    auto &sleef_funs_fx8 = sf::functions::sleef::get_funs_fx8();
    auto &sleef_funs_fx16 = sf::functions::sleef::get_funs_fx16();

    auto &stl_funs_fx1 = sf::functions::stl::get_funs_fx1();
    auto &stl_funs_dx1 = sf::functions::stl::get_funs_dx1();

    std::set<std::string> fun_union;
#define merge_into_set(FUNS)                                                                                           \
    for (auto kv : FUNS)                                                                                               \
        fun_union.insert(kv.first);

    merge_into_set(af_funs_fx8);
    merge_into_set(amdlibm_funs_fx1);
    merge_into_set(boost_funs_fx1);
    merge_into_set(eigen_funs);
    merge_into_set(fort_funs);
    merge_into_set(gsl_funs);
    merge_into_set(misc_funs_cdx1_x2);
    merge_into_set(sctl_funs_fx8);
    merge_into_set(sleef_funs_fx1);
    merge_into_set(stl_funs_fx1);
#undef merge_into_set

    std::set<std::string> keys_to_eval;
    if (input_keys.size() > 0)
        std::set_intersection(fun_union.begin(), fun_union.end(), input_keys.begin(), input_keys.end(),
                              std::inserter(keys_to_eval, keys_to_eval.end()));
    else
        keys_to_eval = fun_union;

    std::vector<std::pair<int, int>> run_sets;
    for (uint8_t shift = 0; shift <= 14; shift += 2)
        run_sets.push_back({1 << (10 + shift), 1 << (14 - shift)});

    std::unordered_map<std::string, configuration_t> base_configurations = {
        {"acos", {.lbound = -1.0, .ubound = 1.0}},
        {"asin", {.lbound = -1.0, .ubound = 1.0}},
        {"atan", {.lbound = -100.0, .ubound = 100.0}},
        {"acosh", {.lbound = 1.0, .ubound = 1000.0}},
        {"asinh", {.lbound = -100.0, .ubound = 100.0}},
        {"atanh", {.lbound = -1.0, .ubound = 1.0}},
        {"bessel_Y0", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_Y1", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_Y2", {.lbound = 0.1, .ubound = 30.0}},
        {"cos_pi", {.lbound = 0.0, .ubound = 2.0}},
        {"sin_pi", {.lbound = 0.0, .ubound = 2.0}},
        {"cos", {.lbound = 0.0, .ubound = 2 * M_PI, .ilbound = 0.0, .iubound = 2 * M_PI}},
        {"sin", {.lbound = 0.0, .ubound = 2 * M_PI, .ilbound = 0.0, .iubound = 2 * M_PI}},
        {"tan", {.lbound = 0.0, .ubound = 2 * M_PI}},
        {"erf", {.lbound = -1.0, .ubound = 1.0}},
        {"erfc", {.lbound = -1.0, .ubound = 1.0}},
        {"exp", {.lbound = -10.0, .ubound = 10.0}},
        {"log", {.lbound = 0.0, .ubound = 10.0}},
        {"hank103", {.lbound = 0.0, .ubound = 10.0, .ilbound = 0.0, .iubound = 10.0}},
    };

    auto &baobzi_funs = sf::functions::baobzi::get_funs_dx1(keys_to_eval, base_configurations);

    for (auto &run_set : run_sets) {
        const auto &[n_eval, n_repeat] = run_set;
        std::cerr << "Running benchmark with input vector of length " << n_eval << " and " << n_repeat << " repeats.\n";
        Eigen::VectorXd vals = 0.5 * (Eigen::ArrayXd::Random(n_eval) + 1.0);
        Eigen::VectorXf fvals = vals.cast<float>();
        Eigen::VectorX<cdouble> cvals = 0.5 * (Eigen::ArrayX<cdouble>::Random(n_eval) + std::complex<double>{1.0, 1.0});

        for (auto key : keys_to_eval) {
            auto conf_f = base_configurations[key];
            conf_f.ftype = "f";
            conf_f.func = key;
            auto insert_measurement = [&storage](measurement_t &meas) -> void {
                if (meas)
                    storage.insert(meas);
            };

            using namespace sqlite_orm;
            auto conf_fids = storage.select(columns(&configuration_t::id),
                                            where(is_equal(&configuration_t::ftype, conf_f.ftype) and
                                                  is_equal(&configuration_t::func, conf_f.func) and
                                                  is_equal(&configuration_t::lbound, conf_f.lbound) and
                                                  is_equal(&configuration_t::ubound, conf_f.ubound) and
                                                  is_equal(&configuration_t::ilbound, conf_f.ilbound) and
                                                  is_equal(&configuration_t::iubound, conf_f.iubound)));
            conf_f.id = conf_fids.size() ? std::get<int>(conf_fids[0]) : storage.insert(conf_f);

            std::vector<measurement_t> ms;
            ms.push_back(test_func(amdlibm_funs_fx1[key], 1, libraries_info["amdlibm"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(amdlibm_funs_fx8[key], 8, libraries_info["amdlibm"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(af_funs_fx8[key], 8, libraries_info["agnerfog"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(af_funs_fx16[key], 16, libraries_info["agnerfog"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(boost_funs_fx1[key], 1, libraries_info["boost"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(eigen_funs[key], 0, libraries_info["eigen"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(sleef_funs_fx1[key], 1, libraries_info["sleef"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(sleef_funs_fx8[key], 8, libraries_info["sleef"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(sleef_funs_fx16[key], 16, libraries_info["sleef"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(sctl_funs_fx8[key], 8, libraries_info["sctl"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(sctl_funs_fx16[key], 16, libraries_info["sctl"], conf_f, fvals, n_repeat));
            ms.push_back(test_func(stl_funs_fx1[key], 1, libraries_info["stl"], conf_f, fvals, n_repeat));

            auto conf_d = base_configurations[key];
            conf_d.func = key;
            conf_d.ftype = "d";
            auto conf_dids = storage.select(columns(&configuration_t::id),
                                            where(is_equal(&configuration_t::ftype, conf_d.ftype) and
                                                  is_equal(&configuration_t::func, conf_d.func) and
                                                  is_equal(&configuration_t::lbound, conf_d.lbound) and
                                                  is_equal(&configuration_t::ubound, conf_d.ubound) and
                                                  is_equal(&configuration_t::ilbound, conf_d.ilbound) and
                                                  is_equal(&configuration_t::iubound, conf_d.iubound)));
            conf_d.id = conf_dids.size() ? std::get<int>(conf_dids[0]) : storage.insert(conf_d);
            ms.push_back(test_func(af_funs_dx4[key], 4, libraries_info["agnerfog"], conf_d, vals, n_repeat));
            ms.push_back(test_func(af_funs_dx8[key], 8, libraries_info["agnerfog"], conf_d, vals, n_repeat));
            ms.push_back(test_func(amdlibm_funs_dx1[key], 1, libraries_info["amdlibm"], conf_d, vals, n_repeat));
            ms.push_back(test_func(amdlibm_funs_dx4[key], 4, libraries_info["amdlibm"], conf_d, vals, n_repeat));
            ms.push_back(test_func(baobzi_funs[key], 1, libraries_info["baobzi"], conf_d, vals, n_repeat));
            ms.push_back(test_func(boost_funs_dx1[key], 1, libraries_info["boost"], conf_d, vals, n_repeat));
            ms.push_back(test_func(eigen_funs[key], 0, libraries_info["eigen"], conf_d, vals, n_repeat));
            ms.push_back(test_func(fort_funs[key], 1, libraries_info["fort"], conf_d, vals, n_repeat));
            ms.push_back(test_func(gsl_funs[key], 1, libraries_info["gsl"], conf_d, vals, n_repeat));
            ms.push_back(test_func(sctl_funs_dx4[key], 4, libraries_info["sctl"], conf_d, vals, n_repeat));
            ms.push_back(test_func(sctl_funs_dx8[key], 8, libraries_info["sctl"], conf_d, vals, n_repeat));
            ms.push_back(test_func(sleef_funs_dx1[key], 1, libraries_info["sleef"], conf_d, vals, n_repeat));
            ms.push_back(test_func(sleef_funs_dx4[key], 4, libraries_info["sleef"], conf_d, vals, n_repeat));
            ms.push_back(test_func(sleef_funs_dx8[key], 8, libraries_info["sleef"], conf_d, vals, n_repeat));
            ms.push_back(test_func(stl_funs_dx1[key], 1, libraries_info["stl"], conf_d, vals, n_repeat));

            for (auto &meas : ms) {
                std::cout << meas;
                insert_measurement(meas);
            }
            // test_func(gsl_complex_funs, [key], "gsl_cdx1", params, cvals, n_repeat);
            // test_func(misc_funs_cdx1_x2[key], "misc_cdx1_x2", params, cvals, n_repeat);

            std::cout << "\n";
        }
    }
    return 0;
}
