"""
进阶·统计套利 ③ Johansen 多资产协整
======================================
[①](../01-协整与配对交易/) 的 Engle-Granger 检验一次只能看【两个】资产。真实世界里,
一篮子(3、5、10 个)资产可能共享更深的协整结构,而且可能有【不止一条】协整关系。

Johansen 检验(1991) 一次性处理【N 个资产】:
  · 检验协整的【秩 r】(有几条独立的、平稳的线性组合);
  · 用特征值分解,直接给出这些【协整向量】(篮子权重),按"多平稳"排序。

  实验①  3 个资产共享 1 条随机趋势→各自随机游走(非平稳),但存在协整篮子
  实验②  Johansen 迹检验(trace test): 正确识别协整秩 r=2(=资产数−共同趋势数)
  实验③  最平稳的协整向量→篮子价差,平稳绕 0(对照单个资产的乱走)
  实验④  交易这个多资产篮子价差: 均值回归, 半衰期 + 夏普

运行：python johansen_cointegration.py
依赖：numpy, statsmodels, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pathlib import Path

RNG = np.random.default_rng(0)
TD = 252


def ou(n, phi=0.92, s=1.0, rng=RNG):
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, s)
    return x


def make_basket(n=600, rng=RNG):
    """3 个资产共享【1 条】随机趋势 + 各自平稳扰动 → 协整秩应为 3−1=2。"""
    trend = np.cumsum(rng.normal(0, 1, n))
    Y = np.column_stack([
        100 + trend + ou(n, rng=rng),
        100 + 1.5 * trend + ou(n, rng=rng),
        100 - 0.8 * trend + ou(n, rng=rng),
    ])
    return Y


def half_life(spread):
    """OU 半衰期:价差回到一半偏离要多久(闭环①)。"""
    s = spread - spread.mean()
    ds = np.diff(s)
    reg = np.polyfit(s[:-1], ds, 1)[0]
    return -np.log(2) / reg if reg < 0 else np.inf


def main():
    print("=" * 60)
    print("  进阶·统计套利 ③ Johansen 多资产协整")
    print("=" * 60)
    Y = make_basket()
    n = len(Y)
    res = coint_johansen(Y, det_order=0, k_ar_diff=1)
    trace, crit95 = res.lr1, res.cvt[:, 1]
    rank = int((trace > crit95).sum())
    print(f"① 3 个资产共享 1 条随机趋势 → 各自非平稳(随机游走)")
    print(f"② Johansen 迹检验: trace={np.round(trace,1)} vs 95%临界={np.round(crit95,1)}")
    print(f"   协整秩 r = {rank}（=资产数3 − 共同趋势数1 = 2,正确识别！）\n")

    vec = res.evec[:, 0]                       # 最平稳的协整向量(篮子权重)
    spread = Y @ vec
    hl = half_life(spread)
    print(f"③ 最平稳协整向量(篮子权重): {np.round(vec,2)}")
    print(f"   篮子价差标准差 {spread.std():.2f}(平稳) vs 单资产 {Y[:,0].std():.1f}(乱走);半衰期 {hl:.0f} 天\n")

    # ④ 交易篮子价差(均值回归)
    z = (spread - spread.mean()) / spread.std()
    pos = np.zeros(n)
    for t in range(1, n):
        p = pos[t - 1]
        if p == 0:
            if z[t] > 1: p = -1
            elif z[t] < -1: p = 1
        elif abs(z[t]) < 0.2:
            p = 0
        pos[t] = p
    dsp = np.diff(spread, append=spread[-1])
    pnl = pos * dsp
    sharpe = pnl.mean() / pnl.std() * np.sqrt(TD) if pnl.std() > 0 else 0
    print(f"④ 交易多资产篮子价差(z>1做空/z<-1做多): 年化夏普 {sharpe:.2f}")
    print("   一篮子比两两配对用了更多信息,价差更稳、更可交易。")
    print("   ⚠️ 这是理想演示(价差按构造平稳、样本内权重、未计成本);真实中协整向量")
    print("      要样本外估、会漂移(用 Kalman②)、会失效(LTCM),夏普远没这么漂亮。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for j, c in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ax.plot(Y[:, j], color=c, lw=0.9, label=f"asset {j+1}")
    ax.set_title("(1) Three assets sharing a common stochastic trend (each non-stationary)")
    ax.set_xlabel("time"); ax.set_ylabel("price"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    r_labels = ["r=0", "r≤1", "r≤2"]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, trace, w, color="#1f77b4", label="trace statistic")
    ax.bar(x + w / 2, crit95, w, color="#d62728", alpha=0.7, label="95% critical")
    ax.set_xticks(x); ax.set_xticklabels(r_labels)
    ax.set_title(f"(2) Johansen trace test → cointegration rank = {rank}")
    ax.set_ylabel("statistic"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for i in range(3):
        mark = "reject→coint" if trace[i] > crit95[i] else "accept"
        ax.text(i, max(trace[i], crit95[i]) + 1, mark, ha="center", fontsize=7.5,
                color="#1f77b4" if trace[i] > crit95[i] else "#888")

    ax = axes[1, 0]
    ax.plot(spread, color="#9467bd", lw=0.9, label="basket spread β'Y")
    ax.axhline(spread.mean(), color="k", lw=0.8)
    ax.axhline(spread.mean() + spread.std(), color="#2ca02c", ls="--", lw=1, label="±1σ bands")
    ax.axhline(spread.mean() - spread.std(), color="#2ca02c", ls="--", lw=1)
    ax.set_title(f"(3) Most-stationary cointegrating basket is mean-reverting (half-life {hl:.0f}d)")
    ax.set_xlabel("time"); ax.set_ylabel("spread"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(np.cumsum(pnl), color="#1f77b4", lw=1.3, label=f"basket pairs-trade  Sharpe={sharpe:.2f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(4) Trading the multi-asset basket spread (market-neutral)")
    ax.set_xlabel("time"); ax.set_ylabel("cumulative P&L (spread units)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "Johansen配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（多资产/迹检验/篮子价差/交易 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
