"""
进阶·编程深化 ② Numba 即时编译（JIT）——介于 Python 和 C++ 之间
==================================================================
[① C++ 课](../01-CPP加速蒙特卡洛/) 的结论:路径依赖+提前退出的问题,C++ 比 NumPy 快;
但写 C++ 要另起一套(头文件、编译、绑定)。Numba 给出"第三条路":

   给一个普通的 Python 循环函数加一行 @njit 装饰器,它就被【即时编译成机器码】,
   速度逼近 C++——而你【一行 C++ 都没写】。

这对"Python + 一点 C++"的你是福音:大多数性能问题,Numba 就够了,根本不用上 C++。

  实验①  价格验证:Python / NumPy / Numba 三法都收敛到 Black-Scholes
  实验②  欧式期权(可向量化):Numba ≈ NumPy(都快),纯Python 慢几十倍
  实验③  障碍期权(路径依赖+提前退出):Numba 像 C++ 一样【碾压】NumPy,但零 C++
  实验④  加速比总览:一行装饰器换来几十倍加速(代码量几乎不变)

运行：python numba_jit.py   (首次运行 Numba 会有一次性 JIT 编译耗时,已预热排除)
依赖：numpy, scipy, numba, matplotlib
"""
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from numba import njit

N = stats.norm.cdf


def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S0 * N(d1) - K * np.exp(-r * T) * N(d1 - sigma * np.sqrt(T))


# ---------- 纯 Python ----------
def euro_python(S0, K, r, sigma, T, n, seed=0):
    rng = np.random.default_rng(seed)
    drift, vol, s = (r - 0.5 * sigma**2) * T, sigma * np.sqrt(T), 0.0
    for _ in range(n):
        p = S0 * np.exp(drift + vol * rng.standard_normal()) - K
        if p > 0:
            s += p
    return np.exp(-r * T) * s / n


def barrier_python(S0, K, B, r, sigma, T, n, steps, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps; drift = (r - 0.5 * sigma**2) * dt; vol = sigma * np.sqrt(dt); s = 0.0
    for _ in range(n):
        S = S0; alive = True
        for _ in range(steps):
            S *= np.exp(drift + vol * rng.standard_normal())
            if S <= B:
                alive = False; break
        if alive and S - K > 0:
            s += S - K
    return np.exp(-r * T) * s / n


# ---------- NumPy 向量化 ----------
def euro_numpy(S0, K, r, sigma, T, n, seed=0):
    rng = np.random.default_rng(seed)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * rng.standard_normal(n))
    return np.exp(-r * T) * np.maximum(ST - K, 0).mean()


def barrier_numpy(S0, K, B, r, sigma, T, n, steps, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps; drift = (r - 0.5 * sigma**2) * dt; vol = sigma * np.sqrt(dt)
    S = np.full(n, float(S0)); alive = np.ones(n, bool)
    for _ in range(steps):                       # 无法对单条路径提前退出→所有步都得走
        S *= np.exp(drift + vol * rng.standard_normal(n))
        alive &= S > B
    return np.exp(-r * T) * np.where(alive, np.maximum(S - K, 0), 0.0).mean()


# ---------- Numba JIT(就是普通 Python 循环 + @njit) ----------
@njit(cache=True)
def euro_numba(S0, K, r, sigma, T, n):
    drift, vol, s = (r - 0.5 * sigma**2) * T, sigma * np.sqrt(T), 0.0
    for _ in range(n):
        p = S0 * np.exp(drift + vol * np.random.standard_normal()) - K
        if p > 0:
            s += p
    return np.exp(-r * T) * s / n


@njit(cache=True)
def barrier_numba(S0, K, B, r, sigma, T, n, steps):
    dt = T / steps; drift = (r - 0.5 * sigma**2) * dt; vol = sigma * np.sqrt(dt); s = 0.0
    for _ in range(n):
        S = S0; alive = True
        for _ in range(steps):
            S *= np.exp(drift + vol * np.random.standard_normal())
            if S <= B:                            # 和 C++ 一样,敲出立刻退出
                alive = False; break
        if alive and S - K > 0:
            s += S - K
    return np.exp(-r * T) * s / n


def timeit(fn, *a, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*a)
    return out, (time.perf_counter() - t0) / reps


def main():
    print("=" * 62)
    print("  进阶·编程深化 ② Numba 即时编译（JIT）")
    print("=" * 62)
    S0, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 1.0
    bs = bs_call(S0, K, r, sigma, T)
    euro_numba(S0, K, r, sigma, T, 10)            # 预热 JIT(把一次性编译耗时排除在计时外)
    barrier_numba(S0, K, 98.0, r, sigma, T, 10, 50)
    print(f"  Black-Scholes 解析解: {bs:.4f}（首次已预热 Numba JIT)\n")

    # ① 验证
    nv = 300_000
    print(f"① 价格验证: Python {euro_python(S0,K,r,sigma,T,40000):.3f} | "
          f"NumPy {euro_numpy(S0,K,r,sigma,T,nv):.3f} | Numba {euro_numba(S0,K,r,sigma,T,nv):.3f} (≈BS {bs:.3f})\n")

    # ② 欧式计时
    n_e = 300_000
    _, tep = timeit(euro_python, S0, K, r, sigma, T, n_e)
    _, ten = timeit(euro_numpy, S0, K, r, sigma, T, n_e, reps=3)
    _, teb = timeit(lambda: euro_numba(S0, K, r, sigma, T, n_e), reps=3)
    print(f"② 欧式(n={n_e:,}): Python {tep*1e3:.0f}ms | NumPy {ten*1e3:.1f}ms | Numba {teb*1e3:.1f}ms")
    print(f"   Numba 比纯Python快 {tep/teb:.0f}×,和 NumPy 同档(可向量化问题大家都快)\n")

    # ③ 障碍期权计时(路径依赖+提前退出)
    B, steps, n_b = 98.0, 250, 80_000
    _, tbp = timeit(barrier_python, S0, K, B, r, sigma, T, n_b, steps)
    _, tbn = timeit(lambda: barrier_numpy(S0, K, B, r, sigma, T, n_b, steps), reps=2)
    _, tbb = timeit(lambda: barrier_numba(S0, K, B, r, sigma, T, n_b, steps), reps=2)
    print(f"③ 障碍(B={B:.0f},steps={steps},n={n_b:,}): Python {tbp*1e3:.0f}ms | NumPy {tbn*1e3:.0f}ms | Numba {tbb*1e3:.0f}ms")
    print(f"   Numba 比 NumPy 快 {tbn/tbb:.1f}×、比纯Python快 {tbp/tbb:.0f}×——像 C++ 一样赢,却零 C++!\n")

    # ④ 加速比
    print(f"④ 一行 @njit,欧式提速 {tep/teb:.0f}×、障碍提速 {tbp/tbb:.0f}×,代码量几乎不变。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ns = np.logspace(2.5, 6, 16).astype(int)
    prices = [euro_numba(S0, K, r, sigma, T, int(nn)) for nn in ns]
    ax.semilogx(ns, prices, "o-", color="#1f77b4", ms=3, label="Numba MC")
    ax.axhline(bs, color="#d62728", ls="--", lw=1.5, label=f"Black-Scholes {bs:.3f}")
    ax.set_title("(1) Numba MC converges to the exact price")
    ax.set_xlabel("paths N"); ax.set_ylabel("call price"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    methods = ["pure\nPython", "NumPy", "Numba\n@njit"]
    te = [tep * 1e3, ten * 1e3, teb * 1e3]
    bars = ax.bar(methods, te, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_yscale("log"); ax.set_title(f"(2) European (vectorizable): Numba ≈ NumPy, n={n_e:,}")
    ax.set_ylabel("time (ms, log)"); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, te):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    tb = [tbp * 1e3, tbn * 1e3, tbb * 1e3]
    bars = ax.bar(methods, tb, color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_yscale("log"); ax.set_title("(3) Barrier (path-dependent): Numba crushes NumPy, like C++")
    ax.set_ylabel("time (ms, log)"); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, tb):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    labels = ["European", "Barrier"]
    sp = [tep / teb, tbp / tbb]
    bars = ax.bar(labels, sp, color=["#1f77b4", "#9467bd"])
    ax.set_title("(4) Numba speedup over pure Python (one decorator!)")
    ax.set_ylabel("speedup ×"); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, sp):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}×", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "Numba配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（收敛/欧式计时/障碍计时/加速比 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
