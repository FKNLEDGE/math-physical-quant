"""
进阶·风险管理 ③ 风险归因与成分 VaR（风险≠资金）
==================================================
[①②](../01-VaR与CVaR/) 度量了组合【总】风险。但风控真正要问的是:**这风险是谁贡献的?**
答案常常反直觉:**一个小仓位的高波动/高相关资产,可能贡献了大部分风险。风险 ≠ 资金。**

  欧拉分解(因为组合波动 σ_p 是权重的一次齐次函数):
    边际风险贡献 MCR_i = ∂σ_p/∂w_i = (Σw)_i / σ_p   (每多配一点 i,总风险增多少)
    成分风险贡献 CCR_i = w_i · MCR_i                  (i 实际占了总风险多少)
    且 Σ CCR_i = σ_p  ——总风险被【干净地】拆给各资产。
  (高斯下,成分 VaR_i = z · CCR_i,同一套分解。)

  实验①  等权组合:权重% vs 风险贡献%——高波动资产的风险远超它的资金占比
  实验②  风险/权重倍数:谁是"风险大户"(>1,该减)、谁是"分散器"(<1)
  实验③  欧拉分解:成分风险堆叠 = 组合总风险(干净可加)
  实验④  等权 vs 风险平价:风险平价把成分风险【拉平】(闭环 C4)

运行：python risk_attribution.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def build_cov(vols, beta, sig_m):
    """单因子结构协方差:共同市场因子 + 各自特异,对角=各资产总方差。"""
    Sigma = np.outer(beta, beta) * sig_m**2
    Sigma += np.diag(vols**2 - np.diag(Sigma))
    return Sigma


def risk_contrib(w, Sigma):
    sp = np.sqrt(w @ Sigma @ w)
    mcr = (Sigma @ w) / sp          # 边际
    ccr = w * mcr                   # 成分(Σ=σ_p)
    return sp, mcr, ccr


def risk_parity(Sigma, iters=8000):
    """求等成分风险(风险平价)权重:w_i ∝ 1/(Σw)_i 的不动点。"""
    d = len(Sigma); w = np.full(d, 1 / d)
    for _ in range(iters):
        w = 1.0 / (Sigma @ w); w /= w.sum()
    return w


def main():
    print("=" * 60)
    print("  进阶·风险管理 ③ 风险归因与成分 VaR（风险≠资金）")
    print("=" * 60)
    names = [f"A{i}" for i in range(6)]
    vols = np.array([0.12, 0.15, 0.18, 0.22, 0.30, 0.40])
    beta = np.array([0.6, 0.8, 1.0, 1.1, 1.3, 1.5])
    Sigma = build_cov(vols, beta, 0.15)
    d = len(vols)
    w = np.full(d, 1 / d)                      # 等权
    sp, mcr, ccr = risk_contrib(w, Sigma)
    pct = ccr / sp
    print(f"  6 资产等权组合;组合波动 σ_p={sp:.1%};成分风险之和={ccr.sum():.1%}(=σ_p,欧拉✓)\n")
    print("① 风险≠资金(等权,每个权重 16.7%):")
    for i in range(d):
        print(f"   {names[i]}(vol{vols[i]:.0%}): 风险贡献 {pct[i]:5.1%}  (风险/权重 {pct[i]/w[i]:.2f}x)")
    print(f"   最高波动的 {names[-1]} 占 {pct[-1]:.0%} 风险却只占 17% 资金——风险藏在它那里。\n")

    wp = risk_parity(Sigma)
    spp, _, ccrp = risk_contrib(wp, Sigma)
    print(f"④ 风险平价权重: {np.round(wp*100,1)}% → 成分风险 {np.round(ccrp/spp*100,1)}%(拉平到≈16.7%)")
    print("   风险平价不按资金等分,而按【风险】等分:高波动资产配得少。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(d)

    ax = axes[0, 0]
    wd = 0.38
    ax.bar(x - wd / 2, w * 100, wd, color="#888", label="weight %")
    ax.bar(x + wd / 2, pct * 100, wd, color="#d62728", label="risk contribution %")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n{v:.0%}" for n, v in zip(names, vols)], fontsize=8)
    ax.set_title("(1) Risk ≠ capital: high-vol assets hog risk, not weight")
    ax.set_ylabel("%"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ratio = pct / w
    ax.bar(x, ratio, color=["#2ca02c" if r < 1 else "#d62728" for r in ratio])
    ax.axhline(1, color="k", lw=1, ls="--", label="risk = weight")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("(2) Risk/weight multiplier: >1 = risk hog (trim), <1 = diversifier")
    ax.set_ylabel("risk% / weight%"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    bottom = 0
    colors = plt.cm.viridis(np.linspace(0, 0.85, d))
    for i in range(d):
        ax.bar(0, ccr[i], bottom=bottom, color=colors[i], label=f"{names[i]} ({pct[i]:.0%})")
        bottom += ccr[i]
    ax.axhline(sp, color="k", lw=1.5, ls="--", label=f"total σ_p = {sp:.1%}")
    ax.set_xlim(-1, 2); ax.set_xticks([0]); ax.set_xticklabels(["portfolio"])
    ax.set_title("(3) Euler decomposition: components stack to total risk")
    ax.set_ylabel("risk contribution"); ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.bar(x - wd / 2, pct * 100, wd, color="#d62728", label="equal-weight")
    ax.bar(x + wd / 2, ccrp / spp * 100, wd, color="#2ca02c", label="risk parity")
    ax.axhline(100 / d, color="k", ls=":", lw=1, label=f"equal risk = {100/d:.1f}%")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("(4) Risk parity equalizes risk contribution (vs lumpy equal-weight)")
    ax.set_ylabel("risk contribution %"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "风险归因配图.png"
    fig.savefig(out, dpi=110)
    print(f"② 风险大户=高波动高相关资产;③ 成分风险干净可加=σ_p;④ 风险平价拉平。")
    print(f"  图已保存: {out.name}（风险vs资金/倍数/欧拉分解/风险平价 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
