"""
A+2 · 端到端实战项目（一个完整、诚实、可复用的研究流程）
=====================================================
把前面学的全串起来,走一遍专业的研究流程,并诚实地下结论:
  数据 → 信号(第04课) → 防前视/扣成本(第05课) → 仓位管理(A+1) → 样本外检验 → 诚实评估

这是一个【模板】:把它的每一步换成你自己的想法,就是你的研究框架。

  实验①  价格 + 信号(均线持仓区间)
  实验②  净值对比:买入持有 vs 原始策略 vs 波动率目标策略(含样本内外分界)
  实验③  回撤对比
  实验④  波动率目标在起作用:仓位随'反波动率'缩放,把策略波动稳定在目标

运行：python end_to_end_project.py
依赖：numpy, pandas, matplotlib（数据用 量化/数据/ 示例行情）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quant_tools as qt

TD = qt.TRADING_DAYS
COST_BPS = 5.0
TARGET_VOL = 0.10        # 波动率目标:让策略年化波动≈10%
MAX_LEV = 1.5            # 仓位上限(防过度杠杆)


def vol_target_size(close, signal):
    """波动率目标仓位:size = 目标波动 / 近期已实现波动(只用过去),并封顶。"""
    ret = qt.to_log_returns(close)
    realized = ret.rolling(20).std() * np.sqrt(TD)        # 近20日年化波动
    size = (TARGET_VOL / realized).shift(1).clip(0, MAX_LEV)  # shift:只用过去
    return (signal * size).reindex(ret.index).fillna(0.0)


def report(name, m):
    print(f"  {name:<16}{m['总收益']:>9.1%}{m['年化收益']:>9.1%}{m['年化波动']:>8.1%}"
          f"{m['夏普比率']:>7.2f}{m['最大回撤']:>9.1%}")


def main():
    print("=" * 64)
    print("  A+2 · 端到端实战项目")
    print("=" * 64)
    close = qt.load_sample_data()["Close"]

    # === 步骤1: 信号(第04课的均线交叉,已内置防前视 shift) ===
    signal = qt.ma_crossover_position(close, 20, 60)   # 0/1 持仓

    # === 步骤2-4: 三种方案的回测(都扣成本) ===
    bt_hold = qt.backtest_long_only(close, pd.Series(1.0, index=close.index), 0)
    bt_raw = qt.backtest_long_only(close, signal, COST_BPS)
    size = vol_target_size(close, signal)              # A+1的仓位管理
    bt_vt = qt.backtest_long_only(close, size, COST_BPS)

    # === 步骤5: 样本外检验(后40%为从未参与设计的样本外) ===
    split = int(len(close) * 0.6)
    split_date = close.index[split]

    print("  策略: 均线20/60信号 + 波动率目标仓位(目标10%,封顶1.5x) + 单边5bp成本")
    print("-" * 64)
    print(f"  {'方案':<16}{'总收益':>9}{'年化':>9}{'波动':>8}{'夏普':>7}{'回撤':>9}")
    print("  【全样本】")
    report("买入持有", qt.perf_metrics(bt_hold["市场收益"]))
    report("原始均线策略", qt.perf_metrics(bt_raw["策略收益"]))
    report("波动率目标策略", qt.perf_metrics(bt_vt["策略收益"]))
    print("  【样本外(后40%,从未用于设计)】")
    report("买入持有(OOS)", qt.perf_metrics(bt_hold["市场收益"].iloc[split:]))
    report("波动目标(OOS)", qt.perf_metrics(bt_vt["策略收益"].iloc[split:]))
    print("-" * 64)

    # === 步骤6: 诚实清单 ===
    realized_vt = bt_vt["策略收益"].std() * np.sqrt(TD)
    print("  ✅ 诚实清单(第05课):")
    print(f"    □ 防前视? 是(信号与仓位都 shift)   □ 扣成本? 是(5bp)")
    print(f"    □ 比基准? 是(买入持有)            □ 样本外? 是(后40%)")
    print(f"    □ 波动率目标达成? 实际年化波动 {realized_vt:.1%} (目标 {TARGET_VOL:.0%})")
    mh = qt.perf_metrics(bt_hold["市场收益"]); mv = qt.perf_metrics(bt_vt["策略收益"])
    print("\n  📌 诚实结论:")
    if mv["夏普比率"] > mh["夏普比率"]:
        print(f"    波动率目标把夏普从买入持有的 {mh['夏普比率']:.2f} 提升到 {mv['夏普比率']:.2f},")
        print(f"    且最大回撤从 {mh['最大回撤']:.0%} 改善到 {mv['最大回撤']:.0%}——靠的是【风险控制】,非择时神力。")
    else:
        print(f"    即便流程严谨,该简单策略夏普({mv['夏普比率']:.2f})仍未超买入持有({mh['夏普比率']:.2f})。")
    print("    这正是专业流程该给你的诚实答案:多数简单想法,做对了也就这样。\n")

    # === 画研究报告仪表盘 ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(close.index, close, color="#333", lw=0.8, label="Close")
    ax.plot(close.index, close.rolling(20).mean(), color="#1f77b4", lw=0.8, label="MA20")
    ax.plot(close.index, close.rolling(60).mean(), color="#ff7f0e", lw=0.8, label="MA60")
    ax.fill_between(close.index, close.min(), close.max(),
                    where=(signal > 0).reindex(close.index).fillna(False),
                    color="#2ca02c", alpha=0.08)
    ax.set_title("(1) Price + MA signal (green = holding)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(bt_hold.index, bt_hold["市场净值"], color="#888", label="buy & hold")
    ax.plot(bt_raw.index, bt_raw["策略净值"], color="#ff7f0e", lw=0.9, label="raw MA")
    ax.plot(bt_vt.index, bt_vt["策略净值"], color="#1f77b4", lw=1.3, label="vol-targeted")
    ax.axvline(split_date, color="red", ls="--", lw=1, label="OOS split")
    ax.set_title("(2) Equity: buy&hold vs raw vs vol-targeted")
    ax.set_ylabel("net value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for bt, c, name in [(bt_hold["市场净值"], "#888", "buy & hold"),
                        (bt_vt["策略净值"], "#1f77b4", "vol-targeted")]:
        dd = bt/bt.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color=c, label=name)
    ax.set_title("(3) Drawdown: vol-targeting cuts the pain")
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(size.index, size.values, color="#9467bd", lw=0.7)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_title("(4) Position size scales with 1/volatility (risk control)")
    ax.set_xlabel("date"); ax.set_ylabel("position size")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "A+2配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（信号/净值/回撤/仓位 四合一）")
    print("=" * 64)


if __name__ == "__main__":
    main()
