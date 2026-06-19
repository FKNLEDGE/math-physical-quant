"""
第 04 课 · 第一个策略：均线交叉
================================
终于要做交易决策了！规则极简：
  - 快均线(20日) 上穿 慢均线(60日)  -> 看涨，满仓持有
  - 快均线 下穿 慢均线            -> 看跌，空仓离场

这节课你会：
  1. 用移动均线生成『买/卖信号』。
  2. 理解为什么信号必须【右移一天】(杜绝前视偏差)。
  3. 做一次『向量化回测』，并【始终和买入持有对比】。
  4. 诚实地解读结果——它真的比躺着不动强吗？

运行：python lesson04_ma_crossover.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quant_tools as qt

FAST, SLOW = 20, 60
COST_BPS = 5.0   # 单边交易成本：5 个基点 = 0.05%（一个偏乐观但合理的估计）


def main():
    df = qt.load_sample_data()
    close = df["Close"]

    # === 1. 生成信号（持仓 0/1）===
    # ma_crossover_position 内部已做 shift(1)：今天收盘才知道均线关系，最早明天才能交易
    position = qt.ma_crossover_position(close, FAST, SLOW)
    fast_ma = close.rolling(FAST).mean()
    slow_ma = close.rolling(SLOW).mean()

    # === 2. 回测：策略 vs 买入持有 ===
    bt = qt.backtest_long_only(close, position, cost_bps=COST_BPS)
    strat_m = qt.perf_metrics(bt["策略收益"])
    hold_m = qt.perf_metrics(bt["市场收益"])

    n_trades = int(position.diff().abs().fillna(0).gt(0).sum())
    time_in_market = position.mean()

    print("=" * 60)
    print(f"  第 04 课 · 均线交叉策略 ({FAST}/{SLOW}, 成本 {COST_BPS:.0f}bp)")
    print("=" * 60)
    print(f"  交易次数: {n_trades}    持仓时间占比: {time_in_market:.0%}（其余时间空仓）")
    print("-" * 60)
    print(f"  {'指标':<10}{'均线策略':>14}{'买入持有':>14}")
    print(f"  {'总收益':<10}{strat_m['总收益']:>14.2%}{hold_m['总收益']:>14.2%}")
    print(f"  {'年化收益':<10}{strat_m['年化收益']:>14.2%}{hold_m['年化收益']:>14.2%}")
    print(f"  {'年化波动':<10}{strat_m['年化波动']:>14.2%}{hold_m['年化波动']:>14.2%}")
    print(f"  {'夏普比率':<10}{strat_m['夏普比率']:>14.2f}{hold_m['夏普比率']:>14.2f}")
    print(f"  {'最大回撤':<10}{strat_m['最大回撤']:>14.2%}{hold_m['最大回撤']:>14.2%}")
    print("-" * 60)
    # 诚实解读
    if strat_m["总收益"] < hold_m["总收益"]:
        print("  📌 诚实结论：策略【跑输】了买入持有的总收益！")
        print(f"     但它把最大回撤从 {hold_m['最大回撤']:.0%} 改善到 {strat_m['最大回撤']:.0%}，")
        print("     即『牺牲部分收益，换取避开大跌』——这是趋势跟踪的典型权衡。")
        print("     ⚠️ 注意：风险调整后的夏普并没有变好，说明这笔交换并不划算。")
    else:
        print("  📌 策略总收益超过买入持有——但先别高兴，第 05 课会拷问它是不是运气。")
    print("  🧠 核心习惯：任何策略都要和一个『笨基准』(买入持有)比，否则毫无意义。")

    # === 3. 画图 ===
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上：价格 + 双均线 + 持仓区间（绿色阴影=满仓）
    ax = axes[0]
    ax.plot(close.index, close, color="#333", lw=1.0, label="Close")
    ax.plot(fast_ma.index, fast_ma, color="#1f77b4", lw=1.0, label=f"MA{FAST}")
    ax.plot(slow_ma.index, slow_ma, color="#ff7f0e", lw=1.0, label=f"MA{SLOW}")
    ax.fill_between(close.index, close.min(), close.max(),
                    where=(position > 0).reindex(close.index).fillna(False),
                    color="#2ca02c", alpha=0.08, label="In market (long)")
    ax.set_title(f"MA Crossover {FAST}/{SLOW}: green = holding")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # 下：净值对比
    ax = axes[1]
    ax.plot(bt.index, bt["策略净值"], color="#2ca02c", lw=1.3, label="MA strategy")
    ax.plot(bt.index, bt["市场净值"], color="#888", lw=1.3, label="Buy & Hold")
    ax.set_title("Equity Curve: strategy vs buy & hold")
    ax.set_ylabel("net value")
    ax.set_xlabel("date")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "第04课配图.png"
    fig.savefig(out, dpi=110)
    print(f"\n  图已保存: {out.name}")


if __name__ == "__main__":
    main()
