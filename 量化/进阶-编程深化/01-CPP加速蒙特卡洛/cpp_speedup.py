"""
进阶·编程深化 ① 用 C++ 给蒙特卡洛定价加速（pybind11）
======================================================
你的工具栈是 Python + 一点 C++。这一课告诉你【什么时候、怎么】上 C++:
不是一上来就写 C++,而是先 Python 跑通、定位热点,再把【最内层的循环】搬到 C++。

同一个蒙特卡洛期权定价([B4 课](../../路线B-衍生品定价/04-蒙特卡洛定价/)),三种写法对比。
【反直觉但重要的结论】:C++ 不是"永远更快"——

  ① 纯 Python 循环  —— 最直观,但最慢(解释器逐元素开销),比下面两个慢几十倍。
  ② NumPy 向量化    —— "Pythonic",一行搞定。它【本身就是在调用 C/SIMD】,
                        所以对【能向量化】的问题,它常常和朴素 C++ 一样快、甚至更快!
  ③ C++ + pybind11  —— 只有当问题【难以向量化】时(路径依赖 + 提前退出、逐路径分支),
                        C++ 才真正赢——因为它能对每条路径独立地提前退出,省下大量计算。

  真正的工程智慧:【先向量化(NumPy),向量化使不上劲时才上 C++】——别过早优化。

  实验①  三种方法都收敛到 Black-Scholes 解析解(先证明"快"不是靠算错)
  实验②  欧式期权(可向量化):NumPy 已经和 C++ 一样快,纯Python 慢几十倍
  实验③  NumPy 和 C++ 相对纯Python 的加速比(两者相当,都碾压纯Python)
  实验④  路径依赖障碍期权(难向量化):大量路径提前敲出时,C++ 才真正甩开 NumPy

运行：python cpp_speedup.py   (首次会自动用 g++ 编译 C++ 扩展)
依赖：numpy, scipy, matplotlib, pybind11, g++（C++17）
"""
import sys
import subprocess
import sysconfig
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
N = stats.norm.cdf


def build_cpp():
    """用 g++ 把 mc_pricer.cpp 编译成可 import 的 mc_cpp 扩展(若需要)。"""
    ext = sysconfig.get_config_var("EXT_SUFFIX")          # 例如 .cpython-311-...-.so
    so = HERE / f"mc_cpp{ext}"
    src = HERE / "mc_pricer.cpp"
    if so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        return
    inc = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"]).decode().split()
    inc += ["-I" + sysconfig.get_path("include")]
    cmd = ["g++", "-O3", "-shared", "-std=c++17", "-fPIC", *inc, str(src), "-o", str(so)]
    print("  编译 C++ 扩展:", " ".join(cmd[:6]), "...")
    subprocess.check_call(cmd)
    print("  ✓ 编译完成\n")


def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * N(d1) - K * np.exp(-r * T) * N(d2)


def mc_python(S0, K, r, sigma, T, n, seed=0):
    """纯 Python 循环(故意写成最朴素的样子)。"""
    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    s = 0.0
    for _ in range(n):
        ST = S0 * np.exp(drift + vol * rng.standard_normal())
        p = ST - K
        if p > 0:
            s += p
    return np.exp(-r * T) * s / n


def mc_numpy(S0, K, r, sigma, T, n, seed=0):
    """NumPy 向量化:一次生成所有路径。"""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    return np.exp(-r * T) * np.maximum(ST - K, 0).mean()


def mc_numpy_barrier(S0, K, B, r, sigma, T, n, steps, seed=0):
    """NumPy 向量化障碍期权:无法'提前退出',必须把所有路径所有步都走完。"""
    rng = np.random.default_rng(seed)
    dt = T / steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    S = np.full(n, float(S0))
    alive = np.ones(n, bool)
    for _ in range(steps):                      # 时间步无法向量化,且不能对单条路径提前退出
        S *= np.exp(drift + vol * rng.standard_normal(n))
        alive &= S > B
    payoff = np.where(alive, np.maximum(S - K, 0), 0.0)
    return np.exp(-r * T) * payoff.mean()


def timeit(fn, *a, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*a)
    return out, (time.perf_counter() - t0) / reps


def main():
    print("=" * 60)
    print("  进阶·编程深化 ① C++ 给蒙特卡洛加速（pybind11）")
    print("=" * 60)
    build_cpp()
    import mc_cpp

    S0, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 1.0
    bs = bs_call(S0, K, r, sigma, T)
    print(f"  Black-Scholes 解析解(标准答案): {bs:.4f}\n")

    # ① 三法都收敛到 BS
    n_val = 400_000
    p_py = mc_python(S0, K, r, sigma, T, 60_000)            # 纯Python用小一点的n(否则太慢)
    p_np = mc_numpy(S0, K, r, sigma, T, n_val)
    p_cpp = mc_cpp.european_call(S0, K, r, sigma, T, n_val, 0)
    print(f"① 价格验证: Python {p_py:.4f} | NumPy {p_np:.4f} | C++ {p_cpp:.4f} (都≈BS {bs:.4f})\n")

    # ② 欧式计时(可向量化):NumPy 已经追平/超过朴素 C++
    n_time = 200_000
    _, t_py = timeit(mc_python, S0, K, r, sigma, T, n_time)
    _, t_np = timeit(mc_numpy, S0, K, r, sigma, T, n_time, reps=3)
    _, t_cpp = timeit(lambda: mc_cpp.european_call(S0, K, r, sigma, T, n_time, 0), reps=3)
    print(f"② 欧式计时(n={n_time:,}): 纯Python {t_py*1e3:.0f}ms | NumPy {t_np*1e3:.1f}ms | C++ {t_cpp*1e3:.1f}ms")
    print(f"   纯Python 比 NumPy 慢 {t_py/t_np:.0f}×;但 NumPy 和朴素 C++ 旗鼓相当"
          f"(NumPy {'更快' if t_np<t_cpp else '稍慢'})——能向量化时,朴素 C++ 白忙。\n")

    # ③ NumPy / C++ 相对纯Python 的加速比(两者都碾压纯Python,且彼此相当)
    ns = [20_000, 50_000, 100_000, 200_000]
    sp_np, sp_cpp = [], []
    for nn in ns:
        _, a = timeit(mc_python, S0, K, r, sigma, T, nn)
        _, b = timeit(lambda: mc_cpp.european_call(S0, K, r, sigma, T, nn, 0), reps=3)
        _, c = timeit(mc_numpy, S0, K, r, sigma, T, nn, reps=3)
        sp_np.append(a / c); sp_cpp.append(a / b)
    print(f"③ 相对纯Python加速比: NumPy {[f'{x:.0f}×' for x in sp_np]} | C++ {[f'{x:.0f}×' for x in sp_cpp]}")
    print("   两者都把纯Python甩开几十倍,而彼此相当——所以【先向量化】是性价比之王。\n")

    # ④ 路径依赖障碍期权:大量路径提前敲出时,C++ 才真正赢
    B = 98.0; steps = 250; n_bar = 80_000     # B 贴近现价→大量路径很早就敲出
    _, tb_np = timeit(lambda: mc_numpy_barrier(S0, K, B, r, sigma, T, n_bar, steps), reps=2)
    pb_cpp, tb_cpp = timeit(lambda: mc_cpp.barrier_call(S0, K, B, r, sigma, T, n_bar, steps, 0), reps=2)
    pb_np = mc_numpy_barrier(S0, K, B, r, sigma, T, n_bar, steps)
    print(f"④ 向下敲出障碍(B={B:.0f}贴近现价, steps={steps}): NumPy {tb_np*1e3:.0f}ms | C++ {tb_cpp*1e3:.0f}ms")
    print(f"   C++ 比NumPy快 {tb_np/tb_cpp:.1f}×。NumPy 必须把所有路径所有步算完,")
    print(f"   C++ 能让敲出的路径【立刻退出】——这才是该上 C++ 的地方。价格 {pb_cpp:.3f}≈{pb_np:.3f}\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 收敛到 BS
    ax = axes[0, 0]
    ns_conv = np.logspace(2.5, 6, 18).astype(int)
    prices = [mc_cpp.european_call(S0, K, r, sigma, T, int(nn), 1) for nn in ns_conv]
    ax.semilogx(ns_conv, prices, "o-", color="#1f77b4", ms=3, label="C++ MC price")
    ax.axhline(bs, color="#d62728", ls="--", lw=1.5, label=f"Black-Scholes {bs:.3f}")
    ax.set_title("(1) All methods converge to the exact price (MC error ~ 1/√N)")
    ax.set_xlabel("paths N"); ax.set_ylabel("call price"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 计时(对数轴)
    ax = axes[0, 1]
    methods = ["pure\nPython", "NumPy", "C++\npybind11"]
    times_ms = [t_py * 1e3, t_np * 1e3, t_cpp * 1e3]
    bars = ax.bar(methods, times_ms, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_yscale("log")
    ax.set_title(f"(2) European is vectorizable: NumPy already matches C++ (n={n_time:,})")
    ax.set_ylabel("time (ms, log)"); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, times_ms):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}ms", ha="center", va="bottom", fontsize=9)

    # 图③ NumPy / C++ 相对纯Python 的加速比(两者相当)
    ax = axes[1, 0]
    ax.plot(ns, sp_np, "s-", color="#ff7f0e", label="NumPy vs pure Python")
    ax.plot(ns, sp_cpp, "o-", color="#2ca02c", label="C++ vs pure Python")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_title("(3) NumPy ≈ C++, both crush pure Python (so: vectorize first)")
    ax.set_xlabel("paths N"); ax.set_ylabel("speedup over pure Python (×)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 障碍期权计时
    ax = axes[1, 1]
    bars = ax.bar(["NumPy\n(all steps)", "C++\n(early exit)"], [tb_np * 1e3, tb_cpp * 1e3],
                  color=["#ff7f0e", "#2ca02c"])
    ax.set_title(f"(4) Hard to vectorize (barrier, early-exit): C++ {tb_np/tb_cpp:.1f}× over NumPy")
    ax.set_ylabel("time (ms)"); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, [tb_np * 1e3, tb_cpp * 1e3]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}ms", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = HERE / "CPP加速配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（收敛/计时/加速比/障碍期权 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
