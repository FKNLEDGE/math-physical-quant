"""
进阶·物理味 ④ 粗糙波动率（Rough Volatility）
================================================
[Heston②](../02-随机波动率Heston/) 让波动率随机,但它的路径还是"标准"扩散(布朗,H=0.5)。
Gatheral 等人 2018《Volatility is Rough》发现一个惊人事实:

   真实的【对数波动率】行为像 Hurst 指数 H≈0.1 的【分数布朗运动】——
   它远比标准布朗运动【粗糙】(锯齿状、反持续/到处均值回归)。

这把 [E2 反常扩散/Hurst](../../路线E-数学物理深化/02-反常扩散Lévy与分形/) 的分形几何,
直接用到了波动率建模,并解释了 Heston 解释不了的【短期波动率微笑骤陡】。

  实验①  分数布朗运动 H=0.1(粗糙) vs 0.5(布朗) vs 0.8(光滑/持续):一眼看出"粗糙"
  实验②  从路径反推 H:增量的 q 阶矩随尺度的标度 m(q,Δ)~Δ^(qH),斜率/2 还原 H
  实验③  单分形签名:ζ_q = qH 是过原点的直线,斜率=H(波动率实测 H≈0.1)
  实验④  后果:ATM 偏斜的期限结构 ~ T^(H−1/2)。H 小→短期偏斜爆炸(正是市场真相)

运行：python rough_volatility.py
依赖：numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from pathlib import Path

RNG = np.random.default_rng(0)


def fbm(H, n, rng=RNG):
    """精确模拟分数布朗运动(Cholesky 法)。H=0.5 即标准布朗运动。"""
    t = np.arange(1, n + 1) / n
    cov = 0.5 * (np.add.outer(t**(2 * H), t**(2 * H)) - np.abs(np.subtract.outer(t, t))**(2 * H))
    L = cholesky(cov + 1e-12 * np.eye(n), lower=True)
    return L @ rng.standard_normal(n)


def scaling_moment(x, q, lags):
    """增量的 q 阶绝对矩 m(q,Δ) = E|x(t+Δ)−x(t)|^q,对一组尺度 Δ。"""
    return np.array([np.mean(np.abs(x[d:] - x[:-d])**q) for d in lags])


def estimate_zeta(x, q, lags):
    """对 log m(q,Δ) ~ ζ_q log Δ 回归,返回标度指数 ζ_q。"""
    m = scaling_moment(x, q, lags)
    return np.polyfit(np.log(lags), np.log(m), 1)[0]


def main():
    print("=" * 60)
    print("  进阶·物理味 ④ 粗糙波动率（Rough Volatility）")
    print("=" * 60)
    n = 2000
    lags = np.arange(1, 50)

    # ① 三种粗糙度的 fBm
    x_rough = fbm(0.1, n, np.random.default_rng(1))
    x_bm = fbm(0.5, n, np.random.default_rng(1))
    x_smooth = fbm(0.8, n, np.random.default_rng(1))

    # ② 从粗糙路径(模拟对数波动率)反推 H
    H_est = estimate_zeta(x_rough, 2, lags) / 2
    H_bm = estimate_zeta(x_bm, 2, lags) / 2
    print(f"① fBm: H=0.1 锯齿粗糙 / H=0.5 标准布朗 / H=0.8 光滑持续")
    print(f"② 从'对数波动率'路径(真H=0.1)反推: m(2,Δ) 斜率/2 = {H_est:.3f}（成功还原粗糙！）")
    print(f"   对照标准布朗(真H=0.5)反推 = {H_bm:.3f}\n")

    # ③ 单分形签名 ζ_q = qH
    qs = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    zetas = np.array([estimate_zeta(x_rough, q, lags) for q in qs])
    H_from_slope = np.polyfit(qs, zetas, 1)[0]
    print(f"③ ζ_q vs q 过原点直线,斜率=H={H_from_slope:.3f}（单分形签名;实测波动率 H≈0.1）\n")

    # ④ ATM 偏斜期限结构 ~ T^(H-1/2)
    T = np.linspace(0.02, 1.0, 100)
    skew_rough = T**(0.1 - 0.5)
    skew_bm = T**(0.5 - 0.5)
    print("④ ATM 偏斜期限结构 ψ(T)~T^(H−1/2):")
    print(f"   H=0.1(粗糙): 短期偏斜随 T→0 像 T^-0.4 爆炸——正是真实市场的样子;")
    print(f"   H=0.5(Heston/布朗): T^0 平坦,短期偏斜不够陡——Heston 的硬伤。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    tt = np.arange(n) / n
    ax.plot(tt, x_smooth, color="#2ca02c", lw=0.8, label="H=0.8 smooth (persistent)")
    ax.plot(tt, x_bm, color="#1f77b4", lw=0.8, label="H=0.5 Brownian")
    ax.plot(tt, x_rough, color="#d62728", lw=0.7, label="H=0.1 ROUGH (vol-like)")
    ax.set_title("(1) Fractional Brownian motion: smaller H = rougher path")
    ax.set_xlabel("time"); ax.set_ylabel("value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for q, c in [(2.0, "#d62728"), (1.0, "#ff7f0e")]:
        m = scaling_moment(x_rough, q, lags)
        ax.plot(np.log(lags), np.log(m), "o", ms=3, color=c, label=f"q={q:.0f} (rough)")
        fit = np.polyfit(np.log(lags), np.log(m), 1)
        ax.plot(np.log(lags), np.polyval(fit, np.log(lags)), "-", color=c, lw=1)
    ax.set_title(f"(2) Moment scaling log m(q,Δ) vs log Δ → H≈{H_est:.2f}")
    ax.set_xlabel("log Δ (scale)"); ax.set_ylabel("log m(q,Δ)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(qs, zetas, "o-", color="#9467bd", label=f"measured ζ_q")
    ax.plot(qs, H_from_slope * qs, "--", color="k", lw=1, label=f"line qH, H={H_from_slope:.2f}")
    ax.set_title("(3) Monofractal signature: ζ_q = qH (straight line through 0)")
    ax.set_xlabel("moment order q"); ax.set_ylabel("scaling exponent ζ_q")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(T, skew_rough, color="#d62728", lw=2, label="rough H=0.1:  ~T^(-0.4)")
    ax.plot(T, skew_bm, color="#1f77b4", lw=2, label="Brownian H=0.5: flat (Heston)")
    ax.set_title("(4) ATM skew term structure ~ T^(H-1/2): rough explodes short-term")
    ax.set_xlabel("maturity T (years)"); ax.set_ylabel("ATM skew (normalized)")
    ax.annotate("market: short-dated\nskew blows up\n→ needs rough vol", xy=(0.05, skew_rough[2]),
                xytext=(0.35, skew_rough[2] * 0.8), fontsize=8.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "粗糙波动率配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（fBm路径/矩标度/单分形/偏斜期限 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
