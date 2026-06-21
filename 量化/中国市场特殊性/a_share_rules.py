"""
中国市场特殊性（A股规则与坑）
============================
美股范式的策略,搬到 A 股常常水土不服。A 股有 T+1、涨跌停、印花税、
散户做空难、停牌等特殊规则——它们直接决定策略可不可行。
【回测必须编码这些真实规则,否则它在对你撒谎。】

  实验①  理想回测 vs A股现实(T+1延迟 + 真实成本):净值的差距
  实验②  指标退化:总收益与夏普被摩擦吃掉多少
  实验③  A股交易成本拆解(佣金/印花税/过户费,买卖不对称)
  实验④  涨跌停:日收益分布与 ±10%/±5%/±20% 限制线

运行：python a_share_rules.py
依赖：numpy, pandas, matplotlib（数据用 量化/数据/ 示例行情）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quant_tools as qt

TD = qt.TRADING_DAYS
# A股单边成本(基点)：佣金~2.5bp + 过户费~0.1bp；卖出额外印花税 5bp(2023年起单边)
BUY_BPS = 2.6
SELL_BPS = 7.6      # 含印花税


def backtest_ashare(close, position, t_plus_1=True):
    """A股现实回测：T+1执行延迟 + 买卖不对称成本(印花税卖出收)。"""
    if t_plus_1:
        position = position.shift(1)        # T+1+执行延迟的近似:再慢一天
    mkt = qt.to_log_returns(close)
    position = position.reindex(mkt.index).fillna(0.0)
    dpos = position.diff().fillna(position)
    buy = dpos.clip(lower=0) * (BUY_BPS/1e4)     # 加仓→买入成本
    sell = (-dpos).clip(lower=0) * (SELL_BPS/1e4) # 减仓→卖出成本(含印花税)
    strat = position*mkt - buy - sell
    return strat, qt.to_equity_curve(strat)


def main():
    print("=" * 60)
    print("  中国市场特殊性（A股规则与坑）")
    print("=" * 60)
    close = qt.load_sample_data()["Close"]
    signal = qt.ma_crossover_position(close, 20, 60)

    # 理想 vs A股现实
    ideal = qt.backtest_long_only(close, signal, cost_bps=0)
    ar_strat, ar_eq = backtest_ashare(close, signal, t_plus_1=True)
    m_ideal = qt.perf_metrics(ideal["策略收益"])
    m_real = qt.perf_metrics(ar_strat)

    print("① 同一个均线策略:")
    print(f"   理想回测(无摩擦):  总收益 {m_ideal['总收益']:.1%}, 夏普 {m_ideal['夏普比率']:.2f}")
    print(f"   A股现实(T+1+成本): 总收益 {m_real['总收益']:.1%}, 夏普 {m_real['夏普比率']:.2f}")
    print(f"   → 摩擦吃掉了 {m_ideal['总收益']-m_real['总收益']:.1%} 的收益。回测不编码规则=自欺。\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 净值差距
    ax = axes[0, 0]
    ax.plot(ideal.index, ideal["策略净值"], color="#2ca02c", label="idealized (no frictions)")
    ax.plot(ar_eq.index, ar_eq, color="#d62728", label="A-share realistic (T+1 + costs)")
    ax.set_title("(1) Idealized vs A-share-realistic equity")
    ax.set_ylabel("net value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 指标退化
    ax = axes[0, 1]
    x = np.arange(2)
    ax.bar(x-0.2, [m_ideal["总收益"]*100, m_ideal["夏普比率"]*10], 0.4, color="#2ca02c", label="idealized")
    ax.bar(x+0.2, [m_real["总收益"]*100, m_real["夏普比率"]*10], 0.4, color="#d62728", label="A-share")
    ax.set_xticks(x); ax.set_xticklabels(["total return (%)", "Sharpe x10"])
    ax.set_title("(2) Metrics degrade under real rules")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 图③ 成本拆解
    ax = axes[1, 0]
    comps = {"commission\n(buy+sell)": 2.6*2, "stamp tax\n(sell only)": 5.0, "transfer\nfee": 0.2}
    ax.bar(comps.keys(), comps.values(), color=["#1f77b4", "#d62728", "#ff7f0e"])
    ax.set_title(f"(3) A-share round-trip cost breakdown (~{2.6*2+5.0+0.2:.0f} bp)")
    ax.set_ylabel("basis points"); ax.grid(alpha=0.3, axis="y")

    # 图④ 涨跌停
    ax = axes[1, 1]
    r = (close.pct_change().dropna()*100)
    ax.hist(r, bins=50, color="#9467bd", edgecolor="white")
    for lim, c, name in [(10, "#d62728", "±10% 主板"), (5, "#ff7f0e", "±5% ST"), (20, "#2ca02c", "±20% 创业板/科创板")]:
        ax.axvline(lim, color=c, ls="--", lw=1); ax.axvline(-lim, color=c, ls="--", lw=1)
    ax.set_title("(4) Price limits (zhang/die ting): can't trade at the limit")
    ax.set_xlabel("daily return (%)"); ax.set_ylabel("count")
    ax.set_xlim(-25, 25); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "A股规则配图.png"
    fig.savefig(out, dpi=110)
    print("② 成本拆解:A股一次往返约 13bp(佣金双边+印花税卖出+过户费)。高换手策略尤其致命。")
    print("③ 涨跌停:主板±10%、ST±5%、创业板/科创板±20%。【封板时往往根本买不到/卖不掉】,")
    print("   回测若假设能在涨停价买入,是严重高估。")
    print("④ 还有:T+1(当天买不能当天卖)、散户做空难(融资融券池小有成本)、停牌可锁死仓位。\n")
    print(f"  图已保存: {out.name}（净值差距/指标退化/成本拆解/涨跌停 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
