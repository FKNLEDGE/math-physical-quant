"""
线性代数 · 组合与协方差的几何（四个实验）
==========================================
线性代数让你『一次处理一篮子资产』。核心是协方差矩阵——风险的结构。

  实验①  分散投资是『唯一的免费午餐』：组合波动随相关性下降
  实验②  马科维茨子弹：相关性越低，风险-收益边界越漂亮
  实验③  相关性的几何：圆形(无关) vs 雪茄形(相关)
  实验④  PCA/特征值：从一堆相关资产里提取『市场因子』

运行：python linear_algebra_portfolio.py
图内文字用英文；讲解在终端与 笔记.md。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(3)


def portfolio_vol(w, sigA, sigB, rho):
    """两资产组合的波动率（标准差）：σ_p = sqrt(wᵀΣw) 的显式展开。"""
    return np.sqrt(w**2 * sigA**2 + (1-w)**2 * sigB**2 + 2*w*(1-w)*rho*sigA*sigB)


def exp1_free_lunch(ax):
    """实验①：组合波动 vs 权重，不同相关性。ρ<1 时出现『分散化下凹』。"""
    sigA = sigB = 0.20
    ws = np.linspace(0, 1, 200)
    for rho, c in [(1.0, "#d62728"), (0.5, "#ff7f0e"), (0.0, "#2ca02c"), (-1.0, "#1f77b4")]:
        ax.plot(ws, [portfolio_vol(w, sigA, sigB, rho) for w in ws],
                color=c, label=f"corr = {rho:+.1f}")
    ax.set_title("(1) Diversification: the only free lunch")
    ax.set_xlabel("weight on asset A")
    ax.set_ylabel("portfolio volatility (risk)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("① 分散投资 = 唯一的免费午餐")
    print(f"   两资产各波动 20%、各占一半：")
    for rho in [1.0, 0.0, -1.0]:
        v = portfolio_vol(0.5, sigA, sigB, rho)
        print(f"     相关性 {rho:+.1f} → 组合波动 {v:.1%}")
    print("   完全正相关(+1)毫无帮助；不相关(0)波动从20%降到14.1%(=20%/√2)；")
    print("   完全负相关(-1)甚至能把风险降到 0！—— 同样收益、更低风险，凭空赚的。\n")


def exp2_markowitz_bullet(ax):
    """实验②：两资产的风险-收益边界（马科维茨子弹），随相关性变化。"""
    muA, muB, sigA, sigB = 0.08, 0.15, 0.15, 0.25
    ws = np.linspace(0, 1, 200)
    rets = ws*muA + (1-ws)*muB
    for rho, c in [(1.0, "#d62728"), (0.3, "#ff7f0e"), (-0.5, "#1f77b4")]:
        vols = [portfolio_vol(w, sigA, sigB, rho) for w in ws]
        ax.plot(vols, rets, color=c, label=f"corr = {rho:+.1f}")
    ax.scatter([sigA, sigB], [muA, muB], color="k", zorder=5)
    ax.annotate("A", (sigA, muA), textcoords="offset points", xytext=(5, 5))
    ax.annotate("B", (sigB, muB), textcoords="offset points", xytext=(5, 5))
    ax.set_title("(2) Markowitz bullet: lower corr = better frontier")
    ax.set_xlabel("risk (volatility)")
    ax.set_ylabel("expected return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("② 马科维茨子弹：把组合画在『风险-收益』平面上")
    print("   相关性越低，曲线越往左凸（同样收益、更低风险）。这就是『有效前沿』的雏形。\n")


def exp3_correlation_geometry(ax):
    """实验③：相关性的几何——散点云的形状。圆=无关，雪茄=相关。"""
    n = 1500
    eA, eB = RNG.standard_normal(n), RNG.standard_normal(n)
    # 无关 ρ≈0
    a0, b0 = eA, eB
    # 相关 ρ=0.85
    rho = 0.85
    a1, b1 = eA, rho*eA + np.sqrt(1-rho**2)*eB
    ax.scatter(a0, b0, s=5, alpha=0.3, color="#1f77b4", label="corr ~ 0 (round)")
    ax.scatter(a1+6, b1, s=5, alpha=0.3, color="#ff7f0e", label="corr = 0.85 (cigar)")
    ax.set_title("(3) Correlation = shape of the cloud")
    ax.set_xlabel("asset A return (shifted for display)")
    ax.set_ylabel("asset B return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    print("③ 相关性的几何：两资产收益的散点云")
    print("   不相关 → 圆形云；强相关 → 雪茄形云。相关性 = 收益向量夹角的余弦(cos)。")
    print(f"   实测：圆云相关 {np.corrcoef(a0,b0)[0,1]:+.2f}，雪茄云相关 {np.corrcoef(a1,b1)[0,1]:+.2f}\n")


def exp4_pca_market_factor(ax):
    """实验④：PCA——从一堆相关资产里提取『市场因子』(最大特征值)。"""
    n_assets, n_days = 8, 1000
    betas = RNG.uniform(0.6, 1.4, n_assets)          # 各资产对市场的敏感度
    market = RNG.standard_normal(n_days)             # 共同的市场因子
    idio = RNG.standard_normal((n_days, n_assets))   # 各自的特异波动
    returns = market[:, None] * betas[None, :] + idio  # r_i = β_i·F + ε_i

    cov = np.cov(returns, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    explained = eigvals / eigvals.sum()

    ax.bar(range(1, n_assets+1), explained, color="#9467bd", edgecolor="white")
    ax.set_title(f"(4) PCA scree: PC1 = 'market factor' ({explained[0]:.0%})")
    ax.set_xlabel("principal component #")
    ax.set_ylabel("variance explained")
    ax.grid(alpha=0.3)
    print("④ PCA / 特征值：8 个都受『同一市场』影响的资产")
    print(f"   协方差矩阵最大特征值占了总方差的 {explained[0]:.0%}——这就是『市场因子』")
    print("   (大盘涨大家一起涨)。其余特征值是各股自己的故事。这是因子模型的雏形。\n")


def text_demo_covariance():
    """文本：协方差矩阵与组合方差 wᵀΣw 的具体计算。"""
    print("【底层】协方差矩阵 Σ 与组合方差 wᵀΣw（两资产、各占一半、不相关）")
    sig = 0.20
    Sigma = np.array([[sig**2, 0.0], [0.0, sig**2]])   # 对角=各自方差，非对角=协方差
    w = np.array([0.5, 0.5])
    var_p = w @ Sigma @ w                               # wᵀΣw
    print(f"   Σ = {np.round(Sigma, 4).tolist()}")
    print(f"   组合方差 = wᵀΣw = {var_p:.4f}，组合波动 = {np.sqrt(var_p):.1%}")
    print(f"   对比：单资产波动 {sig:.0%} → 分散后 {np.sqrt(var_p):.1%}（降了 {(1-np.sqrt(var_p)/sig):.0%}）\n")


def main():
    print("=" * 62)
    print("  线性代数 · 组合与协方差的几何")
    print("=" * 62)
    text_demo_covariance()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    exp1_free_lunch(axes[0, 0])
    exp2_markowitz_bullet(axes[0, 1])
    exp3_correlation_geometry(axes[1, 0])
    exp4_pca_market_factor(axes[1, 1])
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "线性代数配图.png"
    fig.savefig(out, dpi=110)
    print("=" * 62)
    print(f"  图已保存: {out.name}（分散化/有效前沿/相关几何/PCA 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
