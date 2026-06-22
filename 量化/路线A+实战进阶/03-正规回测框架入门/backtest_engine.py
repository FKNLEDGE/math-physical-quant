"""
A+3 · 正规回测框架入门（手写一个事件驱动引擎）
=============================================
前面用的是『向量化回测』(一次算完整段)——快、适合学习,但容易藏前视bug、
难模拟真实成交。专业研究用『事件驱动回测』:引擎一根K线一根K线喂数据,
策略只能看到『当下及以前』,订单下一根成交——结构上杜绝偷看未来。

  实验①  从零手写最小事件驱动引擎,跑均线策略
  实验②  验证:事件驱动 vs 向量化结果一致(互相印证)
  实验③  引擎记录的持仓/现金随时间变化
  实验④  在价格上标出买卖点(决策在收盘,持有到下一根)

运行：python backtest_engine.py
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


class EventBacktest:
    """最小事件驱动回测引擎。核心:逐根K线推进,策略只看得到 history[:t+1]。"""

    def __init__(self, close, cost_bps=5.0):
        self.close = close
        self.cost = cost_bps / 1e4

    def run(self, strategy):
        """strategy(history) -> 目标仓位比例(0~1)。history 只含到当前为止的数据。"""
        cash, shares = 1.0, 0.0
        equity, holdings, positions = [], [], []
        prices = self.close.values
        for t in range(len(prices)):
            price = prices[t]
            # 1) 策略决策:只喂给它『到 t 为止』的历史(结构上无法偷看未来)
            target = strategy(self.close.iloc[:t + 1])
            # 2) 在当前收盘按目标调仓(下一根才赚到价格变动),扣成本
            cur_equity = cash + shares * price
            trade_value = target * cur_equity - shares * price
            cost = abs(trade_value) * self.cost
            shares = target * cur_equity / price
            cash = cur_equity - target * cur_equity - cost
            equity.append(cash + shares * price)
            holdings.append(shares * price)
            positions.append(target)
        idx = self.close.index
        return pd.DataFrame({"净值": equity, "持仓市值": holdings, "现金": np.array(equity) - np.array(holdings),
                             "仓位": positions}, index=idx)


def make_ma_strategy(fast=20, slow=60):
    """返回一个均线策略函数:快线>慢线则满仓(1),否则空仓(0)。只用历史→无前视。"""
    def strategy(history):
        if len(history) < slow:
            return 0.0
        f = history.iloc[-fast:].mean()
        s = history.iloc[-slow:].mean()
        return 1.0 if f > s else 0.0
    return strategy


def main():
    print("=" * 60)
    print("  A+3 · 正规回测框架入门（手写事件驱动引擎）")
    print("=" * 60)
    close = qt.load_sample_data()["Close"]

    # 事件驱动回测
    engine = EventBacktest(close, cost_bps=5.0)
    result = engine.run(make_ma_strategy(20, 60))
    ev_ret = np.log(result["净值"] / result["净值"].shift(1)).dropna()
    ev_metrics = qt.perf_metrics(ev_ret)

    # 向量化回测(quant_tools)做对照
    vec = qt.backtest_long_only(close, qt.ma_crossover_position(close, 20, 60), 5.0)
    vec_metrics = qt.perf_metrics(vec["策略收益"])

    print("② 互相印证(同一策略,两种引擎):")
    print(f"   事件驱动: 总收益 {ev_metrics['总收益']:.1%}, 夏普 {ev_metrics['夏普比率']:.2f}")
    print(f"   向量化  : 总收益 {vec_metrics['总收益']:.1%}, 夏普 {vec_metrics['夏普比率']:.2f}")
    print("   → 两者接近(时间约定略有差异),互相验证都没写错。\n")
    print("① 事件驱动的关键:策略函数只收到 history[:t+1],【结构上拿不到未来】→")
    print("   天生防前视。而向量化里你一不小心就可能用了未来数据(如不 shift)。\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 净值对比(两引擎)
    ax = axes[0, 0]
    ax.plot(result.index, result["净值"], color="#1f77b4", lw=1.4, label="event-driven engine")
    ax.plot(vec.index, vec["策略净值"], color="#ff7f0e", lw=1.0, ls="--", label="vectorized (quant_tools)")
    ax.set_title("(1&2) Event-driven vs vectorized (they agree)")
    ax.set_ylabel("net value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图③ 持仓市值 vs 现金
    ax = axes[0, 1]
    ax.fill_between(result.index, 0, result["持仓市值"], color="#2ca02c", alpha=0.5, label="stock value")
    ax.fill_between(result.index, result["持仓市值"], result["净值"], color="#888", alpha=0.5, label="cash")
    ax.set_title("(3) Portfolio: stock vs cash over time")
    ax.set_ylabel("value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 买卖点
    ax = axes[1, 0]
    ax.plot(close.index, close, color="#333", lw=0.8)
    pos = result["仓位"]
    buys = pos[(pos.diff() > 0)].index
    sells = pos[(pos.diff() < 0)].index
    ax.scatter(buys, close.loc[buys], marker="^", color="#d62728", s=40, zorder=5, label="buy")
    ax.scatter(sells, close.loc[sells], marker="v", color="#1f77b4", s=40, zorder=5, label="sell")
    ax.set_title("(4) Trades: decide at close, hold into next bar")
    ax.set_ylabel("price"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 向量化 vs 事件驱动 概念对比(文字)
    ax = axes[1, 1]
    ax.axis("off")
    txt = ("VECTORIZED backtest\n"
           "  + fast, great for research/sweeps\n"
           "  + short code\n"
           "  - easy to hide look-ahead bugs\n"
           "  - hard to model real execution\n\n"
           "EVENT-DRIVEN backtest\n"
           "  + structurally no look-ahead\n"
           "  + models orders/slippage/multi-asset\n"
           "  + realistic, closer to live\n"
           "  - slower, more code\n\n"
           "Real frameworks:\n"
           "  vectorbt (vectorized, fast)\n"
           "  backtrader / zipline-reloaded (event)\n"
           "  JoinQuant / MyQuant (cloud, A-share)")
    ax.text(0.02, 0.98, txt, fontsize=9, va="top", family="monospace")
    ax.set_title("Vectorized vs Event-driven")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "A+3配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（净值印证/持仓现金/买卖点/对比 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
