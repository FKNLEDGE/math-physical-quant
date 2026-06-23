"""
进阶·物理味 ② 随机波动率（Heston 模型）
========================================
B3 的 Black-Scholes 假设波动率恒定,但真实市场有'波动率微笑'(BS假设破了)。
Heston(1993) 让【波动率自己也是个随机过程】(均值回归),从第一性原理生出
微笑、厚尾、波动聚集——而不像 B3 那样靠人为混合波动率硬造。

  dS = μS dt + √v·S dW₁           (价格)
  dv = κ(θ−v) dt + ξ√v dW₂        (方差,均值回归), dW₁dW₂ = ρ dt

  实验①  一条 Heston 路径:价格 + 它随机起伏的波动率
  实验②  Heston 收益分布比 GBM 厚尾(波动率随机→厚尾)
  实验③  从 Heston 期权价反解隐含波动率→自然涌现【微笑/偏斜】(ρ 控制偏斜)
  实验④  波动聚集:Heston 的 |收益| 自相关>0(GBM 没有)

运行：python heston_stochastic_vol.py
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path

RNG = np.random.default_rng(0)
N = stats.norm.cdf


def heston_paths(S0, r, v0, kappa, theta, xi, rho, T, steps, n_paths):
    """Euler 模拟 Heston(全截断保证方差非负)。返回价格路径与方差路径。"""
    dt = T/steps
    S = np.full((n_paths, steps+1), float(S0))
    v = np.full((n_paths, steps+1), float(v0))
    for t in range(1, steps+1):
        z1 = RNG.standard_normal(n_paths)
        z2 = rho*z1 + np.sqrt(1-rho**2)*RNG.standard_normal(n_paths)
        vp = np.maximum(v[:, t-1], 0)
        v[:, t] = v[:, t-1] + kappa*(theta - vp)*dt + xi*np.sqrt(vp*dt)*z2
        S[:, t] = S[:, t-1]*np.exp((r - 0.5*vp)*dt + np.sqrt(vp*dt)*z1)
    return S, v


def bs_call(S, K, r, sig, T):
    d1 = (np.log(S/K)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T)); d2 = d1-sig*np.sqrt(T)
    return S*N(d1) - K*np.exp(-r*T)*N(d2)


def implied_vol(price, S, K, r, T):
    try:
        return optimize.brentq(lambda s: bs_call(S, K, r, s, T)-price, 1e-3, 4.0)
    except ValueError:
        return np.nan


def main():
    print("=" * 60)
    print("  进阶·物理味 ② 随机波动率（Heston）")
    print("=" * 60)
    S0, r, T = 100.0, 0.02, 1.0
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.5, -0.6
    print(f"  参数: v0={v0}(波动20%), κ={kappa}(回复速度), θ={theta}(长期方差),")
    print(f"        ξ={xi}(波动率的波动), ρ={rho}(杠杆效应:跌时波动升)\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 一条路径:价格 + 随机波动率
    S1, v1 = heston_paths(S0, r, v0, kappa, theta, xi, rho, T, 252, 1)
    ax = axes[0, 0]
    tt = np.linspace(0, T, 253)
    ax.plot(tt, S1[0], color="#1f77b4", label="price")
    ax.set_ylabel("price", color="#1f77b4"); ax.set_xlabel("time")
    ax2 = ax.twinx()
    ax2.plot(tt, np.sqrt(np.maximum(v1[0], 0))*100, color="#d62728", lw=0.9, alpha=0.7, label="vol %")
    ax2.set_ylabel("volatility %", color="#d62728")
    ax.set_title("(1) Heston path: volatility itself is random (mean-reverting)")
    ax.grid(alpha=0.3)
    print("① 波动率不再是常数,而是一条随机起伏、均值回归的曲线(红)。\n")

    # 图② 收益分布:Heston vs GBM
    ax = axes[0, 1]
    Sh, _ = heston_paths(S0, r, v0, kappa, theta, xi, rho, T, 252, 40000)
    ret_h = np.log(Sh[:, -1]/S0)
    ret_g = (r-0.5*theta)*T + np.sqrt(theta*T)*RNG.standard_normal(40000)  # 同均值方差的GBM
    zh = (ret_h-ret_h.mean())/ret_h.std()
    ax.hist(zh, bins=80, density=True, alpha=0.6, color="#1f77b4", label=f"Heston (kurt {stats.kurtosis(ret_h):.1f})")
    xs = np.linspace(-5, 5, 200)
    ax.plot(xs, stats.norm.pdf(xs), "r-", lw=2, label="Gaussian (GBM)")
    ax.set_yscale("log"); ax.set_title("(2) Heston returns are fat-tailed (vs GBM)")
    ax.set_xlabel("standardized return"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"② Heston 收益超额峰度 {stats.kurtosis(ret_h):.1f}>0(厚尾),GBM=0。波动率随机→厚尾。\n")

    # 图③ 隐含波动率微笑/偏斜(ρ 控制)
    ax = axes[1, 0]
    strikes = np.linspace(80, 120, 13)
    for rho_i, c in [(-0.6, "#d62728"), (0.0, "#1f77b4"), (0.6, "#2ca02c")]:
        Sp, _ = heston_paths(S0, r, v0, kappa, theta, xi, rho_i, T, 100, 60000)
        ivs = []
        for K in strikes:
            price = np.exp(-r*T)*np.mean(np.maximum(Sp[:, -1]-K, 0))
            ivs.append(implied_vol(price, S0, K, r, T)*100)
        ax.plot(strikes, ivs, "o-", ms=3, color=c, label=f"ρ={rho_i}")
    ax.axhline(np.sqrt(theta)*100, color="gray", ls="--", lw=0.8, label="BS flat 20%")
    ax.set_title("(3) Implied vol SMILE/SKEW emerges (ρ controls skew)")
    ax.set_xlabel("strike K"); ax.set_ylabel("implied vol %")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("③ 从 Heston 期权价反解隐含波动率,自然涌现【微笑/偏斜】:")
    print("   ρ<0(杠杆)→左高右低的偏斜(怕暴跌,深虚看跌贵),正是股市真实形状。不用人为造!\n")

    # 图④ 波动聚集
    ax = axes[1, 1]
    Slong, _ = heston_paths(S0, r, v0, kappa, theta, xi, rho, 6.0, 1512, 1)
    rh = np.diff(np.log(Slong[0]))
    rg = np.sqrt(theta/252)*RNG.standard_normal(len(rh))
    def acf(x, k=25):
        return [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, k+1)]
    ax.bar(np.arange(1, 26)-0.2, acf(np.abs(rh)), 0.4, color="#d62728", label="Heston |returns|")
    ax.bar(np.arange(1, 26)+0.2, acf(np.abs(rg)), 0.4, color="#888", label="GBM |returns|")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(4) Volatility clustering: Heston |returns| autocorrelated")
    ax.set_xlabel("lag"); ax.set_ylabel("ACF of |returns|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("④ Heston 的 |收益| 自相关>0=波动聚集(大波动扎堆),GBM≈0。闭环时间序列课/E2。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "Heston配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（路径/厚尾/微笑/波动聚集 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
