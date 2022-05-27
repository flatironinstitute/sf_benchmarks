#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include <sys/mman.h>

#include <sf_benchmarks.hpp>
#include <sf_libraries.hpp>
#include <sf_utils.hpp>

#include <sqlite_orm/sqlite_orm.h>
#include <toml.hpp>

run_info_t run_info;
toolchain_info_t toolchain_info;
host_info_t host_info;
std::map<std::string, library_info_t> libraries_info = {
    {"agnerfog", sf::functions::af::library_info},   {"amdlibm", sf::functions::amd::library_info},
    {"baobzi", sf::functions::baobzi::library_info}, {"boost", sf::functions::boost::library_info},
    {"eigen", sf::functions::eigen::library_info},   {"gsl", sf::functions::gsl::library_info},
    {"fort", sf::functions::fort::library_info},     {"misc", sf::functions::misc::library_info},
    {"sctl", sf::functions::SCTL::library_info},     {"sleef", sf::functions::sleef::library_info},
    {"stl", sf::functions::stl::library_info},
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
        os << left << setw(15) << meas.cyclespereval << left << setw(5) << " ";
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

template <typename VAL_T, typename FUN_T>
measurement_t test_func(const FUN_T &f, int veclev, library_info_t &library_info, configuration_t &config,
                        const Eigen::Ref<const Eigen::VectorX<VAL_T>> &x_in,
                        const Eigen::Ref<const Eigen::VectorXd> &y_ref, int n_repeat) {
    if (!f)
        return measurement_t();
    const std::string label = library_info.name + "_" + config.func;

    Eigen::VectorX<VAL_T> x = sf::utils::transform_domain<VAL_T>(x_in, config.lbound, config.ubound);

    size_t res_size = x.size();
    size_t n_evals = x.size() * n_repeat;
    if constexpr (std::is_same_v<FUN_T, fun_cdx1_x2>)
        res_size *= 2;

    Eigen::VectorX<VAL_T> res(res_size);
    // Force virtual memory to RAM (to force malloc to do its thing)
    mlock(res.data(), res_size * sizeof(VAL_T));
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

    meas.run = std::make_unique<int>(run_info.id);
    meas.configuration = std::make_unique<int>(config.id);
    meas.library = std::make_unique<int>(library_info.id);
    meas.nelem = x.size();
    meas.nrepeat = n_repeat;
    meas.cyclespereval = timer.ticks_elapsed() / (double)n_evals;
    meas.megaevalspersec = n_evals / timer.elapsed() / 1E6;
    meas.meanevaltime = timer.elapsed() / n_evals / 1E-9;
    meas.veclev = veclev;

    if (y_ref.size() && (std::is_same_v<VAL_T, float> || std::is_same_v<VAL_T, double>)) {
        Eigen::VectorXd delta = res.template cast<double>() - y_ref;
        meas.maxerr = delta.array().abs().maxCoeff();
        meas.maxrelerr = (delta.array().abs() / y_ref.array().abs()).maxCoeff();
        meas.stddev = std::sqrt((delta.array() - delta.mean()).square().sum() / (delta.size() - 1));

        meas.maxerr = std::isnan(meas.maxerr) ? -2.0 : meas.maxerr;
        meas.maxrelerr = std::isnan(meas.maxrelerr) ? -2.0 : meas.maxrelerr;
        meas.stddev = std::isnan(meas.stddev) ? -2.0 : meas.stddev;
    } else {
        meas.stddev = -1.0;
        meas.maxerr = -1.0;
        meas.maxrelerr = -1.0;
    }

    munlock(res.data(), res_size * sizeof(VAL_T));
    return meas;
}
#undef EIGEN_CASE

template <typename VAL_T, typename FUN_T>
measurement_t test_func_new(const FUN_T &f, int veclev, library_info_t &library_info, configuration_t &config,
                            const Eigen::Ref<const Eigen::VectorX<VAL_T>> &x_in,
                            const Eigen::Ref<const Eigen::VectorXd> &y_ref, int n_repeat) {
    const std::string label = library_info.name + "_" + config.func;

    Eigen::VectorX<VAL_T> x = sf::utils::transform_domain<VAL_T>(x_in, config.lbound, config.ubound);

    size_t res_size = x.size();
    size_t n_evals = x.size() * n_repeat;
    if constexpr (std::is_same_v<FUN_T, fun_cdx1_x2>)
        res_size *= 2;

    Eigen::VectorX<VAL_T> res(res_size);
    // Force virtual memory to RAM (to force malloc to do its thing)
    mlock(res.data(), res_size * sizeof(VAL_T));
    VAL_T *resptr = res.data();

    sf::utils::timer timer;
    for (long k = 0; k < n_repeat; k++)
        f(x.data(), resptr, x.size());
    timer.stop();

    measurement_t meas;
    meas.config_copy = config;
    meas.library_copy = library_info;

    meas.run = std::make_unique<int>(run_info.id);
    meas.configuration = std::make_unique<int>(config.id);
    meas.library = std::make_unique<int>(library_info.id);
    meas.nelem = x.size();
    meas.nrepeat = n_repeat;
    meas.cyclespereval = timer.ticks_elapsed() / (double)n_evals;
    meas.megaevalspersec = n_evals / timer.elapsed() / 1E6;
    meas.meanevaltime = timer.elapsed() / n_evals / 1E-9;
    meas.veclev = veclev;

    if (y_ref.size() && (std::is_same_v<VAL_T, float> || std::is_same_v<VAL_T, double>)) {
        Eigen::VectorXd delta = res.template cast<double>() - y_ref;
        meas.maxerr = delta.array().abs().maxCoeff();
        meas.maxrelerr = (delta.array().abs() / y_ref.array().abs()).maxCoeff();
        meas.stddev = std::sqrt((delta.array() - delta.mean()).square().sum() / (delta.size() - 1));

        meas.maxerr = std::isnan(meas.maxerr) ? -2.0 : meas.maxerr;
        meas.maxrelerr = std::isnan(meas.maxrelerr) ? -2.0 : meas.maxrelerr;
        meas.stddev = std::isnan(meas.stddev) ? -2.0 : meas.stddev;
    } else {
        meas.stddev = -1.0;
        meas.maxerr = -1.0;
        meas.maxrelerr = -1.0;
    }

    munlock(res.data(), res_size * sizeof(VAL_T));
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
        make_table("runs", make_column("id", &run_info_t::id, autoincrement(), primary_key()),
                   make_column("time", &run_info_t::time), make_column("host", &run_info_t::host),
                   make_column("toolchain", &run_info_t::toolchain),
                   foreign_key(&run_info_t::host).references(&host_info_t::id),
                   foreign_key(&run_info_t::toolchain).references(&toolchain_info_t::id)),
        make_table(
            "measurements", make_column("id", &measurement_t::id, autoincrement(), primary_key()),
            make_column("run", &measurement_t::run), make_column("library", &measurement_t::library),
            make_column("configuration", &measurement_t::configuration), make_column("nelem", &measurement_t::nelem),
            make_column("nrepeat", &measurement_t::nrepeat), make_column("veclev", &measurement_t::veclev),
            make_column("megaevalspersec", &measurement_t::megaevalspersec),
            make_column("cyclespereval", &measurement_t::cyclespereval),
            make_column("meanevaltime", &measurement_t::meanevaltime), make_column("stddev", &measurement_t::stddev),
            make_column("maxrelerr", &measurement_t::maxrelerr), make_column("maxerr", &measurement_t::maxerr),
            foreign_key(&measurement_t::run).references(&run_info_t::id),
            foreign_key(&measurement_t::library).references(&library_info_t::id),
            foreign_key(&measurement_t::configuration).references(&configuration_t::id)));

    auto set_id = [&storage](const auto &ids, auto &info) {
        if (ids.size() == 0)
            info.id = storage.insert(info);
        else
            info.id = std::get<int>(ids[0]);
    };

    storage.sync_schema();

    set_id(storage.select(columns(&host_info_t::id), where(is_equal(&host_info_t::cpuname, host_info.cpuname))),
           host_info);
    set_id(storage.select(columns(&toolchain_info_t::id),
                          where(is_equal(&toolchain_info_t::compiler, toolchain_info.compiler) and
                                is_equal(&toolchain_info_t::compilervers, toolchain_info.compilervers) and
                                is_equal(&toolchain_info_t::libcvers, toolchain_info.libcvers))),
           toolchain_info);

    for (auto &[name, lib] : libraries_info) {
        set_id(storage.select(columns(&library_info_t::id), where(is_equal(&library_info_t::name, lib.name) and
                                                                  is_equal(&library_info_t::version, lib.version))),
               lib);
    }

    run_info.time = storage.select(datetime("now")).front();
    run_info.toolchain = std::make_unique<int>(toolchain_info.id);
    run_info.host = std::make_unique<int>(host_info.id);
    run_info.id = storage.insert(run_info);

    return storage;
}

using Storage = decltype(init_storage(""));

int new_main(int argc, char *argv[], Storage &storage) {
    std::cout << host_info.cpuname << std::endl;
    std::cout << "    " + toolchain_info.compiler + ": " + toolchain_info.compilervers << std::endl;
    std::cout << "    libc: " + toolchain_info.libcvers << std::endl;
    for (auto &[key, lib] : libraries_info)
        std::cout << "    " + lib.name + ": " + lib.version << std::endl;

    std::set<std::string> input_keys = parse_args(argc - 1, argv + 1);

    auto &float_funs = sf::functions::get_float_funs();
    auto &double_funs = sf::functions::get_double_funs();
    std::set<std::string> libs;
    std::set<std::string> funcs;
    std::set<int> veclevels;

    for (auto &[key, val] : float_funs) {
        libs.insert(key.lib);
        funcs.insert(key.fun);
        veclevels.insert(key.veclevel);
    }

    std::set<std::string> funcs_to_eval;
    if (input_keys.size() > 0)
        std::set_intersection(funcs.begin(), funcs.end(), input_keys.begin(), input_keys.end(),
                              std::inserter(funcs_to_eval, funcs_to_eval.end()));
    else
        funcs_to_eval = funcs;

    std::vector<std::pair<int, int>> run_sets;
    for (uint8_t shift = 0; shift <= 14; shift += 14)
        run_sets.push_back({1 << (11 + shift), 1 << (14 - shift)});

    auto config_data = toml::parse("../funcs.toml");
    auto domains = toml::find<std::unordered_map<std::string, std::vector<double>>>(config_data, "domains");

    std::unordered_map<std::string, configuration_t> base_configurations;
    for (auto &func : funcs)
        base_configurations[func] = {.func = func, .lbound = domains[func][0], .ubound = domains[func][1]};

    auto get_conf_data = [&storage, &base_configurations](const std::string &name,
                                                          const std::string &ftype) -> configuration_t {
        configuration_t config = base_configurations[name];
        config.func = name;
        config.ftype = ftype;

        using namespace sqlite_orm;
        auto conf_ids =
            storage.select(columns(&configuration_t::id), where(is_equal(&configuration_t::ftype, config.ftype) and
                                                                is_equal(&configuration_t::func, config.func) and
                                                                is_equal(&configuration_t::lbound, config.lbound) and
                                                                is_equal(&configuration_t::ubound, config.ubound) and
                                                                is_equal(&configuration_t::ilbound, config.ilbound) and
                                                                is_equal(&configuration_t::iubound, config.iubound)));
        config.id = conf_ids.size() ? std::get<int>(conf_ids[0]) : storage.insert(config);
        return config;
    };

    for (auto &run_set : run_sets) {
        const auto &[n_eval, n_repeat] = run_set;
        std::cerr << "Running benchmark with input vector of length " << n_eval << " and " << n_repeat << " repeats.\n";
        Eigen::VectorXd dvals = 0.5 * (Eigen::ArrayXd::Random(n_eval) + 1.0);
        Eigen::VectorXf fvals = dvals.cast<float>();
        // Eigen::VectorX<cdouble> cvals = 0.5 * (Eigen::ArrayX<cdouble>::Random(n_eval) +
        // std::complex<double>{1.0, 1.0});

        for (const auto &func : funcs_to_eval) {
            const auto &domain = domains[func];
            Eigen::VectorXd ref;

            for (const auto &stype : std::array<std::string, 2>{"float", "double"}) {
                for (const auto &lib : libs) {
                    for (const auto &veclevel : veclevels) {
                        function_key key = {.lib = lib, .fun = func, .veclevel = veclevel};

                        measurement_t meas;
                        if (stype == "float" && float_funs.count(key)) {
                            auto conf = get_conf_data(key.fun, "f");
                            meas = test_func_new<float>(float_funs[key], key.veclevel, libraries_info[key.lib], conf,
                                                        fvals, ref, n_repeat);
                        }
                        if (stype == "double" && double_funs.count(key)) {
                            auto conf = get_conf_data(key.fun, "d");
                            meas = test_func_new<double>(double_funs[key], key.veclevel, libraries_info[key.lib], conf,
                                                         dvals, ref, n_repeat);
                        }

                        if (meas) {
                            std::cout << meas;
                            storage.insert(meas);
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    Storage storage = init_storage("db.sqlite");

    return new_main(argc, argv, storage);

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

    std::set<std::string> funcs_to_eval;
    if (input_keys.size() > 0)
        std::set_intersection(fun_union.begin(), fun_union.end(), input_keys.begin(), input_keys.end(),
                              std::inserter(funcs_to_eval, funcs_to_eval.end()));
    else
        funcs_to_eval = fun_union;

    std::vector<std::pair<int, int>> run_sets;
    for (uint8_t shift = 0; shift <= 14; shift += 14)
        run_sets.push_back({1 << (11 + shift), 1 << (14 - shift)});

    std::unordered_map<std::string, configuration_t> base_configurations = {
        {"acos", {.lbound = -1.0, .ubound = 1.0}},
        {"acosh", {.lbound = 1.0, .ubound = 1000.0}},
        {"asin", {.lbound = -1.0, .ubound = 1.0}},
        {"asinh", {.lbound = -100.0, .ubound = 100.0}},
        {"atan", {.lbound = -100.0, .ubound = 100.0}},
        {"atanh", {.lbound = -1.0, .ubound = 1.0}},
        {"bessel_I0", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_I1", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_I2", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_J0", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_J1", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_J2", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_K0", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_K1", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_K2", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_Y0", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_Y1", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_Y2", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_j0", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_j1", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_j2", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_y0", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_y1", {.lbound = 0.1, .ubound = 30.0}},
        {"bessel_y2", {.lbound = 0.1, .ubound = 30.0}},
        {"cos", {.lbound = 0.0, .ubound = 2 * M_PI, .ilbound = 0.0, .iubound = 2 * M_PI}},
        {"cos_pi", {.lbound = 0.0, .ubound = 2.0}},
        {"cosh", {.lbound = 0.0, .ubound = 1.0}},
        {"digamma", {.lbound = 0.0, .ubound = 1.0}},
        {"erf", {.lbound = -1.0, .ubound = 1.0}},
        {"erfc", {.lbound = -1.0, .ubound = 1.0}},
        {"exp", {.lbound = -1.0, .ubound = 1.0}},
        {"exp10", {.lbound = -1.0, .ubound = 1.0}},
        {"exp2", {.lbound = -1.0, .ubound = 1.0}},
        {"hank103", {.lbound = 0.0, .ubound = 10.0, .ilbound = 0.0, .iubound = 10.0}},
        {"hermite_0", {.lbound = 0.0, .ubound = 10.0}},
        {"hermite_1", {.lbound = 0.0, .ubound = 10.0}},
        {"hermite_2", {.lbound = 0.0, .ubound = 10.0}},
        {"hermite_3", {.lbound = 0.0, .ubound = 10.0}},
        {"lgamma", {.lbound = 0.0, .ubound = 10.0}},
        {"log", {.lbound = 0.0, .ubound = 10.0}},
        {"log10", {.lbound = 0.0, .ubound = 10.0}},
        {"log2", {.lbound = 0.0, .ubound = 10.0}},
        {"memcpy", {.lbound = 0.0, .ubound = 1.0}},
        {"memset", {.lbound = 0.0, .ubound = 1.0}},
        {"ndtri", {.lbound = 0.0, .ubound = 1.0}},
        {"pow13", {.lbound = 0.0, .ubound = 1.0}},
        {"pow3.5", {.lbound = 0.0, .ubound = 1.0}},
        {"riemann_zeta", {.lbound = 0.0, .ubound = 10.0}},
        {"rsqrt", {.lbound = 0.0, .ubound = 10.0}},
        {"sin", {.lbound = 0.0, .ubound = 2 * M_PI, .ilbound = 0.0, .iubound = 2 * M_PI}},
        {"sin_pi", {.lbound = 0.0, .ubound = 2.0}},
        {"sinc", {.lbound = 0.0, .ubound = 2 * M_PI, .ilbound = 0.0, .iubound = 2 * M_PI}},
        {"sinc_pi", {.lbound = 0.0, .ubound = 2.0}},
        {"sinh", {.lbound = 0.0, .ubound = 2.0}},
        {"sqrt", {.lbound = 0.0, .ubound = 10.0}},
        {"tan", {.lbound = 0.0, .ubound = 2 * M_PI}},
        {"tanh", {.lbound = -1.0, .ubound = 1.0}},
        {"tgamma", {.lbound = -0.0, .ubound = 1.0}},
    };

    std::unordered_map<std::string, multi_eval_func<double>> double_refs = {
        {"acos", stl_funs_dx1["acos"]},
        {"acosh", stl_funs_dx1["acosh"]},
        {"asin", stl_funs_dx1["asin"]},
        {"asinh", stl_funs_dx1["asinh"]},
        {"atan", stl_funs_dx1["atan"]},
        {"atanh", stl_funs_dx1["atanh"]},
        {"bessel_I0", gsl_funs["bessel_I0"]},
        {"bessel_I1", gsl_funs["bessel_I1"]},
        {"bessel_I2", gsl_funs["bessel_I2"]},
        {"bessel_J0", gsl_funs["bessel_J0"]},
        {"bessel_J1", gsl_funs["bessel_J1"]},
        {"bessel_J2", gsl_funs["bessel_J2"]},
        {"bessel_K0", gsl_funs["bessel_K0"]},
        {"bessel_K1", gsl_funs["bessel_K1"]},
        {"bessel_K2", gsl_funs["bessel_K2"]},
        {"bessel_Y0", gsl_funs["bessel_Y0"]},
        {"bessel_Y1", gsl_funs["bessel_Y1"]},
        {"bessel_Y2", gsl_funs["bessel_Y2"]},
        {"bessel_j0", gsl_funs["bessel_j0"]},
        {"bessel_j1", gsl_funs["bessel_j1"]},
        {"bessel_j2", gsl_funs["bessel_j2"]},
        {"bessel_y0", gsl_funs["bessel_y0"]},
        {"bessel_y1", gsl_funs["bessel_y1"]},
        {"bessel_y2", gsl_funs["bessel_y2"]},
        {"memcpy", sctl_funs_dx4["memcpy"]},
        {"cos", stl_funs_dx1["cos"]},
        {"cos_pi", boost_funs_dx1["cos_pi"]},
        {"cosh", stl_funs_dx1["cosh"]},
        {"digamma", boost_funs_dx1["digamma"]},
        {"erf", stl_funs_dx1["erf"]},
        {"erfc", stl_funs_dx1["erfc"]},
        {"exp", stl_funs_dx1["exp"]},
        {"exp10", stl_funs_dx1["exp10"]},
        {"exp2", stl_funs_dx1["exp2"]},
        {"hermite_0", boost_funs_dx1["hermite_0"]},
        {"hermite_1", boost_funs_dx1["hermite_1"]},
        {"hermite_2", boost_funs_dx1["hermite_2"]},
        {"hermite_3", boost_funs_dx1["hermite_3"]},
        {"lgamma", gsl_funs["lgamma"]},
        {"log", stl_funs_dx1["log"]},
        {"log10", stl_funs_dx1["log10"]},
        {"log2", stl_funs_dx1["log2"]},
        {"pow13", stl_funs_dx1["pow13"]},
        {"pow3.5", stl_funs_dx1["pow3.5"]},
        {"riemann_zeta", gsl_funs["riemann_zeta"]},
        {"rsqrt", stl_funs_dx1["rsqrt"]},
        {"sin", stl_funs_dx1["sin"]},
        {"sin_pi", boost_funs_dx1["sin_pi"]},
        {"sinc", gsl_funs["sinc"]},
        {"sinc_pi", gsl_funs["sinc_pi"]},
        {"sinh", stl_funs_dx1["sinh"]},
        {"sqrt", stl_funs_dx1["sqrt"]},
        {"tan", stl_funs_dx1["tan"]},
        {"tanh", stl_funs_dx1["tanh"]},
        {"tgamma", stl_funs_dx1["tgamma"]},
    };

    for (auto key : funcs_to_eval)
        std::cout << key << std::endl;

    auto &baobzi_funs = sf::functions::baobzi::get_funs_dx1(funcs_to_eval, base_configurations);

    for (auto &run_set : run_sets) {
        const auto &[n_eval, n_repeat] = run_set;
        std::cerr << "Running benchmark with input vector of length " << n_eval << " and " << n_repeat << " repeats.\n";
        Eigen::VectorXd vals = 0.5 * (Eigen::ArrayXd::Random(n_eval) + 1.0);
        Eigen::VectorXf fvals = vals.cast<float>();
        Eigen::VectorX<cdouble> cvals = 0.5 * (Eigen::ArrayX<cdouble>::Random(n_eval) + std::complex<double>{1.0, 1.0});

        for (auto fname : funcs_to_eval) {
            auto get_conf_data = [&storage, &base_configurations](const std::string &name,
                                                                  const std::string &ftype) -> configuration_t {
                configuration_t config = base_configurations[name];
                config.func = name;
                config.ftype = ftype;

                using namespace sqlite_orm;
                auto conf_ids = storage.select(columns(&configuration_t::id),
                                               where(is_equal(&configuration_t::ftype, config.ftype) and
                                                     is_equal(&configuration_t::func, config.func) and
                                                     is_equal(&configuration_t::lbound, config.lbound) and
                                                     is_equal(&configuration_t::ubound, config.ubound) and
                                                     is_equal(&configuration_t::ilbound, config.ilbound) and
                                                     is_equal(&configuration_t::iubound, config.iubound)));
                config.id = conf_ids.size() ? std::get<int>(conf_ids[0]) : storage.insert(config);
                return config;
            };

            Eigen::VectorXd vals_ref = sf::utils::transform_domain<double>(vals, base_configurations[fname].lbound,
                                                                           base_configurations[fname].ubound);

            Eigen::VectorXd dref;
            if (double_refs.count(fname)) {
                dref.resize(vals_ref.size());
                double_refs[fname](vals_ref.data(), dref.data(), vals_ref.size());
            }

            std::vector<measurement_t> ms;
            auto &libs = libraries_info;

            auto conf_f = get_conf_data(fname, "f");
            ms.push_back(test_func<float>(amdlibm_funs_fx1[fname], 1, libs["amdlibm"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(amdlibm_funs_fx8[fname], 8, libs["amdlibm"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(af_funs_fx8[fname], 8, libs["agnerfog"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(af_funs_fx16[fname], 16, libs["agnerfog"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(boost_funs_fx1[fname], 1, libs["boost"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(eigen_funs[fname], 0, libs["eigen"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(sleef_funs_fx1[fname], 1, libs["sleef"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(sleef_funs_fx8[fname], 8, libs["sleef"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(sleef_funs_fx16[fname], 16, libs["sleef"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(sctl_funs_fx8[fname], 8, libs["sctl"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(sctl_funs_fx16[fname], 16, libs["sctl"], conf_f, fvals, dref, n_repeat));
            ms.push_back(test_func<float>(stl_funs_fx1[fname], 1, libs["stl"], conf_f, fvals, dref, n_repeat));

            auto conf_d = get_conf_data(fname, "d");
            ms.push_back(test_func<double>(af_funs_dx4[fname], 4, libs["agnerfog"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(af_funs_dx8[fname], 8, libs["agnerfog"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(amdlibm_funs_dx1[fname], 1, libs["amdlibm"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(amdlibm_funs_dx4[fname], 4, libs["amdlibm"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(baobzi_funs[fname], 1, libs["baobzi"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(boost_funs_dx1[fname], 1, libs["boost"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(eigen_funs[fname], 0, libs["eigen"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(fort_funs[fname], 1, libs["fort"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(gsl_funs[fname], 1, libs["gsl"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(sctl_funs_dx4[fname], 4, libs["sctl"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(sctl_funs_dx8[fname], 8, libs["sctl"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(sleef_funs_dx1[fname], 1, libs["sleef"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(sleef_funs_dx4[fname], 4, libs["sleef"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(sleef_funs_dx8[fname], 8, libs["sleef"], conf_d, vals, dref, n_repeat));
            ms.push_back(test_func<double>(stl_funs_dx1[fname], 1, libs["stl"], conf_d, vals, dref, n_repeat));

            for (auto &meas : ms) {
                if (!meas)
                    continue;
                std::cout << meas;
                storage.insert(meas);
            }
            // test_func(gsl_complex_funs, [key], "gsl_cdx1", params, cvals, n_repeat);
            // test_func(misc_funs_cdx1_x2[key], "misc_cdx1_x2", params, cvals, n_repeat);

            std::cout << "\n";
        }
    }
    return 0;
}
