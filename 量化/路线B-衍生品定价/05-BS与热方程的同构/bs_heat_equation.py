"""
B5 · Black-Scholes 就是热方程（你的物理高光时刻）
=================================================
Black-Scholes 偏微分方程，换元后【就是热传导方程】∂u/∂τ = ∂²u/∂x²。
期权定价 = 把到期收益当作『初始温度分布』，让它沿时间『扩散/抹平』。

  实验①  热核：高斯分布随时间 τ 变宽（扩散的基本解）
  实验②  数值解热方程：一个『热斑』随时间扩散、抹平（你数理方法里的经典）
  实验③  把期权收益『扩散』= Black-Scholes：收益曲棍球杆被高斯热核抹成 BS 曲线
  实验④  热核(对数价格里的高斯) = 风险中性密度(价格里的对数正态)

运行：python bs_heat_equation.py
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

Ncdf = stats.norm.cdf


def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*Ncdf(d1) - K*np.exp(-r*T)*Ncdf(d2)


def price_by_diffusion(S0, K, r, sigma, T):
    """把期权价当作『收益 与 风险中性(对数正态=热核)密度 的卷积』数值算出。
    这等价于『让到期收益沿时间扩散』——正是热方程的解。"""
    grid = np.linspace(1e-6, S0*np.exp(8*sigma*np.sqrt(T)) + 4*K, 20000)
    m = np.log(S0) + (r - 0.5*sigma**2)*T          # 风险中性下 ln S_T 的均值
    s = sigma*np.sqrt(T)
    pdf = np.exp(-(np.log(grid) - m)**2/(2*s**2)) / (grid*s*np.sqrt(2*np.pi))
    payoff = np.maximum(grid - K, 0.0)             # 到期收益(初始温度)
    return np.exp(-r*T) * np.trapezoid(payoff*pdf, grid)


def solve_heat_fd(u0, dx, n_steps, dtau):
    """显式有限差分解热方程 ∂u/∂τ=∂²u/∂x²，返回若干时间切片。"""
    lam = dtau/dx**2          # 稳定性要求 lam ≤ 0.5
    u = u0.copy()
    snaps = [u.copy()]
    for k in range(n_steps):
        u[1:-1] = u[1:-1] + lam*(u[2:] - 2*u[1:-1] + u[:-2])
        if (k+1) % (n_steps//4) == 0:
            snaps.append(u.copy())
    return snaps, lam


def main():
    print("=" * 62)
    print("  B5 · Black-Scholes 就是热方程")
    print("=" * 62)
    print("  Black-Scholes 偏微分方程：")
    print("    ∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S − rV = 0")
    print("  令 S=K·e^x, τ=½σ²(T−t), 并提出一个指数因子后，化为：")
    print("    ∂u/∂τ = ∂²u/∂x²   ← 这就是热传导/扩散方程！")
    print("  含义：到期收益 = 初始温度分布；时间倒流(τ增大) = 让它扩散抹平 = 今天的期权价。\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 热核：高斯随 τ 变宽
    ax = axes[0, 0]
    x = np.linspace(-4, 4, 400)
    for tau, c in [(0.05, "#1f77b4"), (0.2, "#2ca02c"), (0.5, "#ff7f0e"), (1.0, "#d62728")]:
        ax.plot(x, np.exp(-x**2/(4*tau))/np.sqrt(4*np.pi*tau), color=c, label=f"tau={tau}")
    ax.set_title("(1) Heat kernel: Gaussian widens with time tau")
    ax.set_xlabel("x"); ax.set_ylabel("u(x, tau)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 数值解热方程：热斑扩散
    ax = axes[0, 1]
    xg = np.linspace(-6, 6, 241); dx = xg[1]-xg[0]
    u0 = np.where(np.abs(xg) < 1.0, 1.0, 0.0)      # 初始：一块『热斑』(方波)
    snaps, lam = solve_heat_fd(u0, dx, n_steps=400, dtau=0.4*dx**2)
    for i, snap in enumerate(snaps):
        ax.plot(xg, snap, label=("initial" if i == 0 else f"after diffusion {i}"))
    ax.set_title(f"(2) Heat equation: a hot spot diffuses & smooths (lam={lam:.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("temperature")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 图③ 期权收益『扩散』= Black-Scholes
    ax = axes[1, 0]
    K, r, sigma = 100.0, 0.05, 0.20
    S0grid = np.linspace(60, 140, 120)
    ax.plot(S0grid, np.maximum(S0grid-K, 0), "k--", lw=1.5, label="payoff at expiry (T=0)")
    for T, c in [(0.25, "#2ca02c"), (1.0, "#1f77b4")]:
        diff = [price_by_diffusion(s, K, r, sigma, T) for s in S0grid]
        ax.plot(S0grid, diff, color=c, lw=2, label=f"diffused payoff, T={T}")
        ax.scatter(S0grid[::12], bs_call(S0grid[::12], K, r, sigma, T),
                   color=c, marker="o", s=25, zorder=5)
    ax.set_title("(3) Diffusing the payoff = Black-Scholes (dots=BS formula)")
    ax.set_xlabel("stock price S"); ax.set_ylabel("call value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 热核(对数价格高斯) = 风险中性密度(价格对数正态)
    ax = axes[1, 1]
    S0 = 100.0
    Sg = np.linspace(20, 250, 600)
    for T, c in [(0.25, "#2ca02c"), (1.0, "#1f77b4")]:
        m = np.log(S0) + (r - 0.5*sigma**2)*T; s = sigma*np.sqrt(T)
        pdf = np.exp(-(np.log(Sg)-m)**2/(2*s**2)) / (Sg*s*np.sqrt(2*np.pi))
        ax.plot(Sg, pdf, color=c, label=f"risk-neutral density, T={T}")
    ax.axvline(S0, color="gray", ls="--", lw=0.8, label="today S0=100")
    ax.set_title("(4) Heat kernel in log-price = lognormal in price")
    ax.set_xlabel("future stock price S_T"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "B5配图.png"
    fig.savefig(out, dpi=110)

    # 数值验证：扩散法 == BS 公式
    print("③ 验证『把收益扩散 = BS 公式』(S0=K=100, r=5%, σ=20%)：")
    for T in [0.25, 1.0]:
        d = price_by_diffusion(100, 100, 0.05, 0.20, T)
        b = bs_call(100, 100, 0.05, 0.20, T)
        print(f"   T={T}: 扩散法 {d:.4f} vs BS公式 {b:.4f}  (差 {abs(d-b):.2e}) ✅")
    print("\n④ 物理大一统：傅里叶(1822热传导) = Bachelier(1900股价) = 爱因斯坦(1905扩散) = 同一方程")
    print("   期权价 = 到期收益 与 高斯热核 的卷积。BS 公式里的 N(·) 正态，就是这个高斯的积分。")
    print(f"\n  图已保存: {out.name}（热核/热扩散/收益扩散=BS/风险中性密度 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
