"""
第 03 课 · 收益率与风险的语言
==============================
价格本身不好比较（100→110 和 10→11 涨幅一样，但价差不同）。
量化用『收益率』说话，并用一套指标描述『风险』。这节课你会：
  1. 区分『简单收益率』和『对数收益率』，理解为什么常用 log。
  2. 亲手算出年化收益、年化波动率、夏普比率、最大回撤、胜率。
  3. 画出『净值曲线』和『水下回撤图』——量化报告的标配。

运行：python lesson03_returns_risk.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quant_tools as qt

TD = qt.TRADING_DAYS  # 252


def main():
    df = qt.load_sample_data()
    close = df["Close"]

    # === 1. 两种收益率 ===
    simple_ret = close.pct_change().dropna()              # 简单收益率 P_t/P_{t-1} - 1
    log_ret = np.log(close / close.shift(1)).dropna()     # 对数收益率 ln(P_t/P_{t-1})

    print("=" * 58)
    print("  第 03 课 · 收益率与风险的语言")
    print("=" * 58)
    print("① 简单收益率 vs 对数收益率（前 3 天）")
    for d in range(3):
        print(f"   {simple_ret.index[d].date()}  简单 {simple_ret.iloc[d]:+.4%}   "
              f"对数 {log_ret.iloc[d]:+.4%}")
    print("   两者在小幅波动时几乎相等；对数收益率的好处是【可加】：")
    total_by_sum = np.exp(log_ret.sum()) - 1
    total_by_price = close.iloc[-1] / close.iloc[0] - 1
    print(f"   把每天对数收益加起来再还原 = {total_by_sum:.2%}")
    print(f"   直接用首尾价格算总收益    = {total_by_price:.2%}   （完全一致 ✅）")

    # === 2. 亲手算风险/收益指标（公式就在眼前）===
    ann_return = log_ret.mean() * TD                 # 年化收益 = 日均收益 × 252
    ann_vol = log_ret.std() * np.sqrt(TD)            # 年化波动 = 日波动 × √252（方差随时间线性增长，故开根号）
    sharpe = ann_return / ann_vol                    # 夏普 = 单位风险换来的收益（这里无风险利率=0）
    equity = qt.to_equity_curve(log_ret)             # 净值曲线
    mdd = qt.max_drawdown(equity)                    # 最大回撤
    win = (log_ret > 0).mean()                       # 胜率（上涨天数占比）

    print("\n② 关键指标（买入持有这只示例股）")
    print(f"   年化收益率 : {ann_return:8.2%}   赚钱速度")
    print(f"   年化波动率 : {ann_vol:8.2%}   颠簸程度（风险）")
    print(f"   夏普比率   : {sharpe:8.2f}   每 1 单位风险换来多少收益（>1 算不错）")
    print(f"   最大回撤   : {mdd:8.2%}   从最高点跌下来最痛的一次")
    print(f"   胜率       : {win:8.1%}   上涨天数占比")
    print("   💡 提醒：高收益常伴随高波动和大回撤。光看收益率会骗人，")
    print("      必须同时看『风险』——这正是量化比『感觉』高明的地方。")

    # 用工具箱里打包好的函数核对（02–05 课都复用它，避免重复造轮子）
    assert abs(qt.perf_metrics(log_ret)["夏普比率"] - sharpe) < 1e-9

    # === 3. 画图：净值 + 水下回撤 + 收益分布 ===
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    axes[0].plot(equity.index, equity.values, color="#1f77b4")
    axes[0].set_title("Equity Curve (Buy & Hold, start=1.0)")
    axes[0].set_ylabel("net value")
    axes[0].grid(alpha=0.3)

    drawdown = (equity / equity.cummax()) - 1
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.5)
    axes[1].set_title(f"Underwater (Drawdown), Max = {mdd:.1%}")
    axes[1].set_ylabel("drawdown")
    axes[1].grid(alpha=0.3)

    axes[2].hist(log_ret.values, bins=50, color="#2ca02c", edgecolor="white")
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set_title("Daily Log-Returns Distribution (note the fat tails)")
    axes[2].set_xlabel("daily log return")
    axes[2].set_ylabel("frequency")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "第03课配图.png"
    fig.savefig(out, dpi=110)
    print(f"\n   图已保存: {out.name}（净值 / 水下回撤 / 收益分布）")


if __name__ == "__main__":
    main()
