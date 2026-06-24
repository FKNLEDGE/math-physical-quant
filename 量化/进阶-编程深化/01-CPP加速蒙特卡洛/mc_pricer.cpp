// 进阶·编程深化 ① 用 C++ 给蒙特卡洛定价加速(pybind11 绑定)
// 编译由同目录的 cpp_speedup.py 自动完成。两个函数:
//   european_call : 欧式看涨的蒙特卡洛(对照 B4 蒙特卡洛定价课)
//   barrier_call  : 向下敲出障碍期权(路径依赖!一旦跌破障碍就提前退出——C++ 的主场)
#include <pybind11/pybind11.h>
#include <random>
#include <cmath>

namespace py = pybind11;

// 欧式看涨:终值一步到位,GBM 解析解
double european_call(double S0, double K, double r, double sigma, double T,
                     long n_paths, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);
    double sum = 0.0;
    for (long i = 0; i < n_paths; ++i) {
        double ST = S0 * std::exp(drift + vol * N(gen));
        double payoff = ST - K;
        if (payoff > 0.0) sum += payoff;
    }
    return std::exp(-r * T) * sum / static_cast<double>(n_paths);
}

// 向下敲出看涨:必须逐步走完路径,一旦 S<=B 就敲出(提前 break)。
// 这种"带提前退出的路径依赖"正是向量化最尴尬、而 C++ 最自然的场景。
double barrier_call(double S0, double K, double B, double r, double sigma, double T,
                    long n_paths, int steps, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    double dt = T / steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol = sigma * std::sqrt(dt);
    double sum = 0.0;
    for (long i = 0; i < n_paths; ++i) {
        double S = S0;
        bool alive = true;
        for (int t = 0; t < steps; ++t) {
            S *= std::exp(drift + vol * N(gen));
            if (S <= B) { alive = false; break; }   // 敲出→这条路径作废,立刻退出
        }
        if (alive) {
            double payoff = S - K;
            if (payoff > 0.0) sum += payoff;
        }
    }
    return std::exp(-r * T) * sum / static_cast<double>(n_paths);
}

PYBIND11_MODULE(mc_cpp, m) {
    m.doc() = "Monte-Carlo option pricers in C++ (pybind11)";
    m.def("european_call", &european_call,
          py::arg("S0"), py::arg("K"), py::arg("r"), py::arg("sigma"),
          py::arg("T"), py::arg("n_paths"), py::arg("seed"));
    m.def("barrier_call", &barrier_call,
          py::arg("S0"), py::arg("K"), py::arg("B"), py::arg("r"), py::arg("sigma"),
          py::arg("T"), py::arg("n_paths"), py::arg("steps"), py::arg("seed"));
}
