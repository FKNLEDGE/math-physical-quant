"""
C1 · 马科维茨有效前沿（组合管理的起点）
=======================================
不预测涨跌，而是科学配置：给定一堆资产，钱怎么分才能『风险-收益最优』？
全程用到线性代数(wᵀΣw) + 最优化(带约束求最小)。

  实验①  随机组合云 + 有效前沿：所有组合的边界，就是『最优』
  实验②  加无风险资产 → 资本市场线 → 切点组合(最大夏普)
  实验③  最优权重长什么样：最大夏普 vs 最小方差 vs 等权(1/N)
  实验④  脆弱性：预期收益稍有扰动，『最优』权重就剧烈乱跳(优化=放大器)

运行：python markowitz_frontier.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(1)

# 四个资产：预期年化收益 μ、波动率 σ、相关矩阵 → 协方差矩阵 Σ
mu = np.array([0.08, 0.12, 0.15, 0.10])
sig = np.array([0.15, 0.20, 0.28, 0.18])
corr = np.array([
    [1.00, 0.30, 0.20, 0.40],
    [0.30, 1.00, 0.50, 0.30],
    [0.20, 0.50, 1.00, 0.25],
    [0.40, 0.30, 0.25, 1.00],
])
Sigma = np.outer(sig, sig) * corr
RF = 0.03   # 无风险利率


def port_stats(w):
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    return ret, vol


def frontier(targets):
    """对每个目标收益，解析解最小方差组合(允许做空)。"""
    inv = np.linalg.inv(Sigma); ones = np.ones(len(mu))
    A = ones @ inv @ ones; B = ones @ inv @ mu; C = mu @ inv @ mu
    D = A*C - B**2
    vols = []
    for mp in targets:
        var = (A*mp**2 - 2*B*mp + C) / D
        vols.append(np.sqrt(var))
    return np.array(vols)


def gmv_weights():
    inv = np.linalg.inv(Sigma); ones = np.ones(len(mu))
    return inv @ ones / (ones @ inv @ ones)


def tangency_weights(mu_vec, rf=RF):
    inv = np.linalg.inv(Sigma)
    w = inv @ (mu_vec - rf)
    return w / w.sum()


def main():
    print("=" * 62)
    print("  C1 · 马科维茨有效前沿")
    print("=" * 62)
    print("  组合收益 = wᵀμ，组合风险 = √(wᵀΣw)（闭环线性代数课）")
    print("  目标：给定风险求最大收益（或给定收益求最小风险）→ 一条『有效前沿』\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 随机组合云 + 有效前沿 + 单个资产
    ax = axes[0, 0]
    W = RNG.dirichlet(np.ones(4), size=4000)        # 4000 个随机(只做多)组合
    rets = W @ mu; vols = np.sqrt(np.einsum("ij,jk,ik->i", W, Sigma, W))
    sharpe = (rets - RF) / vols
    sc = ax.scatter(vols, rets, c=sharpe, s=6, cmap="viridis", alpha=0.5)
    targets = np.linspace(0.07, 0.16, 100)
    ax.plot(frontier(targets), targets, "r-", lw=2, label="efficient frontier")
    ax.scatter(sig, mu, color="black", marker="D", s=40, zorder=5, label="single assets")
    for i in range(4):
        ax.annotate(f"A{i+1}", (sig[i], mu[i]), textcoords="offset points", xytext=(6, 0), fontsize=8)
    ax.set_title("(1) Random portfolios + efficient frontier")
    ax.set_xlabel("risk (volatility)"); ax.set_ylabel("expected return")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Sharpe", fraction=0.046, pad=0.04)

    # 图② 资本市场线 + 切点组合
    ax = axes[0, 1]
    wt = tangency_weights(mu); rt, vt = port_stats(wt)
    ax.plot(frontier(targets), targets, "r-", lw=2, label="efficient frontier")
    xline = np.linspace(0, 0.30, 50)
    ax.plot(xline, RF + (rt-RF)/vt*xline, "b--", lw=1.5, label="Capital Market Line")
    ax.scatter([0], [RF], color="green", s=60, zorder=5, label=f"risk-free {RF:.0%}")
    ax.scatter([vt], [rt], color="red", marker="*", s=200, zorder=5,
               label=f"tangency (max Sharpe={ (rt-RF)/vt:.2f})")
    ax.set_xlim(0, 0.30); ax.set_ylim(0.02, 0.17)
    ax.set_title("(2) Add risk-free -> CML -> tangency (max Sharpe)")
    ax.set_xlabel("risk (volatility)"); ax.set_ylabel("expected return")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"② 切点组合(最大夏普)：夏普 = {(rt-RF)/vt:.3f}，收益 {rt:.1%}，风险 {vt:.1%}")
    print(f"   资本市场线：无风险资产 + 切点组合的任意搭配（两基金分离定理）\n")

    # 图③ 三种组合的权重
    ax = axes[1, 0]
    w_gmv = gmv_weights(); w_eq = np.ones(4)/4
    x = np.arange(4); bw = 0.25
    ax.bar(x-bw, wt, bw, label="max Sharpe", color="#d62728")
    ax.bar(x, w_gmv, bw, label="min variance", color="#1f77b4")
    ax.bar(x+bw, w_eq, bw, label="equal 1/N", color="#2ca02c")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"A{i+1}" for i in range(4)])
    ax.set_title("(3) Optimal weights: max-Sharpe vs min-var vs 1/N")
    ax.set_ylabel("weight"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"③ 最大夏普权重 = {np.round(wt,2)}（可能有做空/极端配置）")
    print(f"   最小方差权重 = {np.round(w_gmv,2)}；等权 = {np.round(w_eq,2)}\n")

    # 图④ 脆弱性：扰动预期收益，最优权重剧烈乱跳
    ax = axes[1, 1]
    many = []
    for _ in range(300):
        mu_noisy = mu + RNG.normal(0, 0.02, 4)      # 预期收益加 2% 噪声
        many.append(tangency_weights(mu_noisy))
    many = np.array(many)
    ax.boxplot(many, tick_labels=[f"A{i+1}" for i in range(4)])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(4) Fragility: tiny return-error -> wild weight swings")
    ax.set_ylabel("max-Sharpe weight"); ax.grid(alpha=0.3)
    print("④ 脆弱性：给预期收益加 2% 的小噪声，最大夏普权重就大幅乱跳(箱线图很宽)")
    print(f"   例：A3 权重范围约 [{many[:,2].min():.1f}, {many[:,2].max():.1f}]")
    print("   —— 这就是『马科维茨之谜』：理论最优对输入(尤其预期收益)极敏感，实务中很脆弱。")
    print("   呼应微积分课『优化是放大器』+ 第05课『过拟合』。难怪等权 1/N 常常实战更稳。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "C1配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（组合云前沿/资本市场线/权重/脆弱性 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
