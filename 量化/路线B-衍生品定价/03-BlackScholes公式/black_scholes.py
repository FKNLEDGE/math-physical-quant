"""
B3 · Black-Scholes 公式（皇冠上的明珠）
=======================================
二叉树取极限(N→∞)就得到 Black-Scholes 闭式公式。这节课把它讲透：

  实验①  期权价值 = 内在价值 + 时间价值：随到期临近，曲线塌向『曲棍球杆』
  实验②  希腊字母 Delta/Gamma：期权对股价的一阶/二阶敏感度
  实验③  希腊字母 Vega/Theta：对波动率/时间的敏感度
  实验④  隐含波动率『微笑/偏斜』：真实市场厚尾 → BS 假设的裂缝

运行：python black_scholes.py
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path

N = stats.norm.cdf       # 标准正态累积分布
n = stats.norm.pdf       # 标准正态密度
RNG = np.random.default_rng(5)


def d1d2(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return d1, d1 - sigma*np.sqrt(T)


def bs_call(S, K, r, sigma, T):
    d1, d2 = d1d2(S, K, r, sigma, T)
    return S*N(d1) - K*np.exp(-r*T)*N(d2)


def bs_put(S, K, r, sigma, T):
    d1, d2 = d1d2(S, K, r, sigma, T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


def greeks(S, K, r, sigma, T):
    d1, d2 = d1d2(S, K, r, sigma, T)
    return {
        "Delta": N(d1),                                       # ∂C/∂S
        "Gamma": n(d1)/(S*sigma*np.sqrt(T)),                  # ∂²C/∂S²
        "Vega":  S*n(d1)*np.sqrt(T)/100,                      # ∂C/∂σ（每 1% 波动）
        "Theta": (-S*n(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*N(d2))/365,  # 每天
        "Rho":   K*T*np.exp(-r*T)*N(d2)/100,                  # 每 1% 利率
    }


def implied_vol(price, S, K, r, T):
    """从市场价反解出 BS 隐含波动率（在 BS 公式上求根）。"""
    try:
        return optimize.brentq(lambda s: bs_call(S, K, r, s, T) - price, 1e-4, 5.0)
    except ValueError:
        return np.nan


def main():
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    print("=" * 62)
    print("  B3 · Black-Scholes 公式")
    print("=" * 62)
    c = bs_call(S0, K, r, sigma, T); p = bs_put(S0, K, r, sigma, T)
    print(f"  C = S·N(d1) - K·e^(-rT)·N(d2)")
    print(f"  例(S=K=100, r=5%, σ=20%, T=1)：看涨 C = {c:.4f}，看跌 P = {p:.4f}")
    print(f"  （正是 B2 二叉树 N→∞ 收敛到的 10.45 ✅）")
    d1, d2 = d1d2(S0, K, r, sigma, T)
    print(f"  N(d2) = {N(d2):.4f} = 风险中性下『到期实值(S_T>K)』的概率")
    print(f"  N(d1) = {N(d1):.4f} = 期权的 Delta（也用于算实值时股票的期望现值）\n")
    print("  五个希腊字母(敏感度，全是导数——闭环微积分课)：")
    for k, v in greeks(S0, K, r, sigma, T).items():
        print(f"    {k:6} = {v:+.4f}")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    Sgrid = np.linspace(60, 140, 300)

    # 图① 期权价值 = 内在 + 时间价值
    ax = axes[0, 0]
    for Tt, c_ in [(1.0, "#1f77b4"), (0.5, "#2ca02c"), (0.1, "#ff7f0e")]:
        ax.plot(Sgrid, bs_call(Sgrid, K, r, sigma, Tt), color=c_, label=f"T={Tt}")
    ax.plot(Sgrid, np.maximum(Sgrid-K, 0), "k--", lw=1.5, label="T=0 (intrinsic)")
    ax.set_title("(1) Value = intrinsic + time value (decays as T->0)")
    ax.set_xlabel("stock price S"); ax.set_ylabel("call value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② Delta & Gamma
    ax = axes[0, 1]
    g = greeks(Sgrid, K, r, sigma, T)
    ax.plot(Sgrid, g["Delta"], color="#1f77b4", label="Delta = dC/dS")
    ax.plot(Sgrid, g["Gamma"]*20, color="#d62728", label="Gamma x20")
    ax.axvline(K, color="gray", ls="--", lw=0.8)
    ax.set_title("(2) Delta (S-shaped 0->1) & Gamma (peaks ATM)")
    ax.set_xlabel("stock price S"); ax.set_ylabel("greek")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图③ Vega & Theta
    ax = axes[1, 0]
    ax.plot(Sgrid, g["Vega"], color="#2ca02c", label="Vega (per 1% vol)")
    ax.plot(Sgrid, g["Theta"]*10, color="#9467bd", label="Theta x10 (per day)")
    ax.axvline(K, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("(3) Vega (vol sens.) & Theta (time decay)")
    ax.set_xlabel("stock price S"); ax.set_ylabel("greek")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 隐含波动率『微笑』：真实厚尾 → 各行权价反解出的 σ 不是平的
    ax = axes[1, 1]
    M = 600000
    # 风险中性的厚尾终值：波动率混合(80%低+20%高)，对称肥尾；构造成鞅 E[S_T]=S0 e^{rT}
    hi = RNG.random(M) < 0.20
    vol = np.where(hi, 0.40, 0.15)
    X = RNG.normal(0.0, vol*np.sqrt(T))
    ST = S0*np.exp(r*T) * np.exp(X) / np.mean(np.exp(X))
    strikes = np.linspace(80, 120, 17)
    ivs = []
    for Kk in strikes:
        price = np.exp(-r*T) * np.mean(np.maximum(ST - Kk, 0))
        ivs.append(implied_vol(price, S0, Kk, r, T))
    ax.plot(strikes, np.array(ivs)*100, "o-", color="#d62728", label="implied vol (smile)")
    ax.axhline(20, color="gray", ls="--", label="flat-vol BS = 20%")
    ax.set_title("(4) Implied volatility SMILE: fat tails break flat-vol BS")
    ax.set_xlabel("strike K"); ax.set_ylabel("implied vol (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "B3配图.png"
    fig.savefig(out, dpi=110)
    print("④ 隐含波动率：从市场价反解出的 σ。若市场真是『常波动率 BS 世界』，它该是平的(20%)。")
    print("   但真实收益厚尾(大涨大跌都比正态多) → 两侧行权价的期权都更贵 → 反解 σ 两头翘 = 『微笑』")
    print("   —— 这是 BS『波动率恒定』假设最著名的裂缝；股市里常呈左高右低的『偏斜』(怕暴跌)。\n")
    print(f"  图已保存: {out.name}（价值/Delta-Gamma/Vega-Theta/隐含波动率 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
