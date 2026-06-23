"""
进阶·统计套利 ① 协整与配对交易
================================
方向难测(时间序列课),但两个'绑在一起'的资产,它们的【价差】会均值回归。
统计套利就赌这个:不赌方向,赌价差回归。核心数学是【协整】。

  实验①  协整对:两个各自随机游走的价格,却'同步'(被橡皮筋拴住)
  实验②  价差:协整对的价差是平稳的(均值回归),非协整对的价差是随机游走
  实验③  协整检验(Engle-Granger) + OU 半衰期
  实验④  配对策略:协整对能赚钱,非协整对不能

运行：python cointegration_pairs.py
依赖：numpy, pandas, statsmodels, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import coint, adfuller

RNG = np.random.default_rng(7)
N = 1000


def ou_spread(n, theta=0.05, sigma=1.0):
    """OU 过程:均值回归的平稳价差。theta=回复速度,半衰期=ln2/theta。"""
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t-1]*(1-theta) + sigma*RNG.standard_normal()
    return s


def hedge_ratio(y, x):
    """OLS 对冲比率 β：y ≈ α + βx。"""
    beta = np.polyfit(x, y, 1)[0]
    return beta


def pairs_pnl(spread, lookback=40):
    """配对策略:价差的滚动z-score,高了做空价差、低了做多(均值回归),shift防前视。"""
    z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    pos = (-z.shift(1)).clip(-2, 2)
    pnl = (pos * spread.diff()).fillna(0)
    sharpe = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() > 0 else 0
    return pnl.cumsum(), sharpe, z


def main():
    print("=" * 60)
    print("  进阶·统计套利 ① 协整与配对交易")
    print("=" * 60)

    # 协整对:共同随机趋势 + 平稳价差
    common = 100 + np.cumsum(0.5*RNG.standard_normal(N))      # 共同的随机游走
    spread_true = ou_spread(N, theta=0.04)                    # 平稳价差(OU)
    A = pd.Series(common)
    B = pd.Series(20 + 1.5*common + 4*spread_true)            # B 与 A 协整(对冲比1.5)
    # 非协整对:A 与一条独立随机游走
    C = pd.Series(150 + np.cumsum(0.5*RNG.standard_normal(N)))

    # 协整检验
    p_coint = coint(A, B)[1]
    p_noncoint = coint(A, C)[1]
    beta = hedge_ratio(B.values, A.values)
    spread_co = B - beta*A
    spread_no = C - hedge_ratio(C.values, A.values)*A
    # OU 半衰期：回归 Δs_t = -θ·s_{t-1} + 噪声，半衰期=ln2/θ
    reg = pd.concat([spread_co.diff().rename("ds"), spread_co.shift(1).rename("lag")], axis=1).dropna()
    theta_est = -np.polyfit(reg["lag"], reg["ds"], 1)[0]
    half_life = np.log(2)/theta_est if theta_est > 0 else np.nan

    print(f"③ 协整检验(Engle-Granger p值,<0.05=协整):")
    print(f"   协整对 A-B : p={p_coint:.4f}  →  {'✅ 协整' if p_coint<0.05 else '不协整'}")
    print(f"   非协整 A-C : p={p_noncoint:.4f}  →  {'协整' if p_noncoint<0.05 else '❌ 不协整'}")
    print(f"   对冲比率 β={beta:.2f}; 价差ADF p={adfuller(spread_co)[1]:.4f}(<0.05=平稳)")
    print(f"   价差均值回归半衰期 ≈ {half_life:.0f} 天\n")

    # 配对策略
    pnl_co, sh_co, z_co = pairs_pnl(spread_co)
    pnl_no, sh_no, _ = pairs_pnl(spread_no)
    print(f"④ 配对策略夏普: 协整对 {sh_co:.2f}  vs  非协整对 {sh_no:.2f}")
    print("   → 协整对的价差均值回归,策略赚钱;非协整对价差是随机游走,策略无效。\n")
    print("⚠️ 风险:协整会【失效】(橡皮筋断裂)。价差可能在回归前先爆走→需止损/限仓。")
    print("   LTCM 正是死于'收敛交易'不收敛(详见简史)。\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 协整对(各自归一化叠加)
    ax = axes[0, 0]
    ax.plot((A-A.mean())/A.std(), color="#1f77b4", label="A (norm)")
    ax.plot((B-B.mean())/B.std(), color="#ff7f0e", label="B (norm)", alpha=0.8)
    ax.set_title("(1) Cointegrated pair: move together (rubber band)")
    ax.set_xlabel("time"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 价差:协整(平稳) vs 非协整(随机游走)
    ax = axes[0, 1]
    ax.plot(spread_co, color="#2ca02c", label="cointegrated spread (stationary)")
    ax.plot(spread_no - spread_no.mean(), color="#d62728", alpha=0.6, label="non-coint spread (random walk)")
    ax.axhline(spread_co.mean(), color="k", ls=":", lw=0.8)
    ax.set_title("(2) Spread: stationary (tradeable) vs random walk (not)")
    ax.set_xlabel("time"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图③ 协整价差 + z-score 交易带
    ax = axes[1, 0]
    m = spread_co.rolling(40).mean(); sd = spread_co.rolling(40).std()
    ax.plot(spread_co, color="#2ca02c", lw=0.8)
    ax.plot(m, color="k", lw=0.8, label="rolling mean")
    ax.fill_between(spread_co.index, m-2*sd, m+2*sd, color="gray", alpha=0.2, label="±2σ band")
    ax.set_title(f"(3) Trade the spread: enter at ±2σ (half-life~{half_life:.0f}d)")
    ax.set_xlabel("time"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 策略累计盈亏
    ax = axes[1, 1]
    ax.plot(pnl_co, color="#2ca02c", label=f"cointegrated (Sharpe {sh_co:.1f})")
    ax.plot(pnl_no, color="#d62728", label=f"non-cointegrated (Sharpe {sh_no:.1f})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(4) Pairs strategy: cointegration is the edge")
    ax.set_xlabel("time"); ax.set_ylabel("cumulative PnL"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "协整配对配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（协整对/价差/交易带/策略盈亏 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
