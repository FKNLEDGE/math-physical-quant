"""
C3 · 因子模型（CAPM 的多因子推广）
==================================
CAPM 用『一个因子(市场)』解释收益，现实需要『多个因子』。
因子模型把收益拆成『对若干共同风格因子的暴露 + 特异』。

  实验①  多因子解释力更强：CAPM 把『价值暴露』误当成 Alpha；3因子还原真相
  实验②  风格暴露：价值/成长/小盘 基金的因子载荷(Beta)长相不同
  实验③  因子动物园：测一堆随机『假因子』，总有几个碰巧显著(呼应第05课)
  实验④  业绩归因：把组合收益拆成 市场+规模+价值+真Alpha

运行：python factor_models.py
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

RNG = np.random.default_rng(11)
TD = 252
N = 2500

# 三个因子的(年化溢价, 年化波动)：市场 MKT、规模 SMB、价值 HML
FAC = {"MKT": (0.06, 0.16), "SMB": (0.02, 0.08), "HML": (0.03, 0.10)}


def sim_factors():
    f = {}
    for k, (prem, vol) in FAC.items():
        f[k] = RNG.normal(prem/TD, vol/np.sqrt(TD), N)
    return f


def sim_stock(f, b_mkt, b_smb, b_hml, alpha=0.0, idio=0.20):
    eps = RNG.normal(alpha/TD, idio/np.sqrt(TD), N)
    return b_mkt*f["MKT"] + b_smb*f["SMB"] + b_hml*f["HML"] + eps


def regress(y, X):
    """最小二乘：y = X·b + 截距。返回 (alpha年化, betas, R²)。"""
    Xd = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ b
    r2 = 1 - resid.var()/y.var()
    return b[0]*TD, b[1:], r2


def main():
    print("=" * 62)
    print("  C3 · 因子模型")
    print("=" * 62)
    print("  R_i - rf = α + β_mkt·MKT + β_smb·SMB + β_hml·HML + ε")
    print("  每个因子=一种『风格收益』，β=对该风格的暴露。CAPM 是只有 MKT 的特例。\n")
    f = sim_factors()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① CAPM 把价值暴露误当 Alpha；3因子还原
    ax = axes[0, 0]
    value_stock = sim_stock(f, b_mkt=1.0, b_smb=0.2, b_hml=0.8, alpha=0.0, idio=0.05)  # 分散化的价值组合,真Alpha=0
    a1, _, r2_1 = regress(value_stock, f["MKT"])                              # CAPM
    a3, _, r2_3 = regress(value_stock, np.column_stack([f["MKT"], f["SMB"], f["HML"]]))
    ax.bar([0, 1], [r2_1, r2_3], color=["#888", "#2ca02c"], width=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"CAPM\nalpha={a1:.1%}", f"3-factor\nalpha={a3:.1%}"])
    ax.set_ylabel("R^2 (variance explained)")
    ax.set_title("(1) 3-factor explains more; spurious 'alpha' vanishes")
    ax.grid(alpha=0.3, axis="y")
    print(f"① 一只真Alpha=0的『价值股』：")
    print(f"   CAPM(单因子): R²={r2_1:.2f}, 估出 alpha={a1:+.1%} ← 把价值暴露误当成『本事』！")
    print(f"   3因子:        R²={r2_3:.2f}, 估出 alpha={a3:+.1%} ← 真相:没有Alpha,只是价值暴露\n")

    # 图② 不同风格基金的因子载荷
    ax = axes[0, 1]
    funds = {
        "Value": sim_stock(f, 1.0, 0.1, 0.8),
        "Growth": sim_stock(f, 1.1, 0.0, -0.6),
        "SmallCap": sim_stock(f, 1.0, 0.9, 0.1),
    }
    x = np.arange(3); bw = 0.25
    loadings = {name: regress(r, np.column_stack([f["MKT"], f["SMB"], f["HML"]]))[1] for name, r in funds.items()}
    for i, fac in enumerate(["MKT", "SMB", "HML"]):
        ax.bar(x + (i-1)*bw, [loadings[name][i] for name in funds], bw, label=fac)
    ax.set_xticks(x); ax.set_xticklabels(list(funds.keys()))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(2) Factor exposures reveal a fund's 'style'")
    ax.set_ylabel("factor loading (beta)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    print("② 风格暴露：价值基金 HML 载荷高，成长基金 HML 为负，小盘基金 SMB 高。")
    print("   因子载荷=基金的『DNA』，一眼看穿它在赌什么风格。\n")

    # 图③ 因子动物园：随机假因子也能碰巧显著
    ax = axes[1, 0]
    target = sim_stock(f, 1.0, 0.0, 0.0)            # 一只普通股
    tstats = []
    for _ in range(300):
        fake = RNG.normal(0, 0.01, N)               # 纯随机的『假因子』
        b = np.cov(target, fake)[0, 1]/np.var(fake)
        se = target.std()/(np.sqrt(N)*fake.std())
        tstats.append(b/se)
    tstats = np.array(tstats)
    n_sig = int((np.abs(tstats) > 1.96).sum())
    ax.hist(tstats, bins=30, color="#ff7f0e", edgecolor="white")
    ax.axvline(1.96, color="red", ls="--"); ax.axvline(-1.96, color="red", ls="--", label="|t|=1.96")
    ax.set_title(f"(3) Factor zoo: {n_sig}/300 random 'factors' look significant")
    ax.set_xlabel("t-stat of a random factor"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"③ 因子动物园：测 300 个【纯随机】假因子，{n_sig} 个碰巧 |t|>1.96 显著(≈5%)。")
    print("   学术界发表了数百个『因子』，大多是这样挖出来的噪声(数据窥探,呼应第05课)。\n")

    # 图④ 业绩归因：把组合收益拆开
    ax = axes[1, 1]
    port = sim_stock(f, 1.0, 0.3, 0.5, alpha=0.02, idio=0.04)   # 分散化基金,含一点真Alpha(+2%)
    a, betas, _ = regress(port, np.column_stack([f["MKT"], f["SMB"], f["HML"]]))
    contrib = {
        "MKT": betas[0]*FAC["MKT"][0],
        "SMB": betas[1]*FAC["SMB"][0],
        "HML": betas[2]*FAC["HML"][0],
        "Alpha": a,
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.bar(list(contrib.keys()), [v*100 for v in contrib.values()], color=colors)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"(4) Attribution: total return {sum(contrib.values())*100:.1f}% split by source")
    ax.set_ylabel("annual return contribution (%)"); ax.grid(alpha=0.3, axis="y")
    print("④ 业绩归因：把组合年化收益拆成 市场/规模/价值/真Alpha 的贡献。")
    print(f"   各部分(%/年)：{ {k: round(float(v*100),1) for k,v in contrib.items()} }")
    print("   —— 一个『跑赢』的基金，多半赢在因子暴露(可廉价复制)，真Alpha往往很小。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "C3配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（解释力/风格暴露/因子动物园/业绩归因 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
