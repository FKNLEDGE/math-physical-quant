"""
A+4 · 从回测到模拟盘（把"学"接到"练"）
======================================
回测过了≠能赚钱。上真钱前还有一道关:【模拟盘(paper trading)】——
用实时数据、真实规则,但不用真钱,做一次"彩排"。它能抓住回测抓不到的问题:
实时数据/延迟、你自己的心理、实盘代码bug、真实滑点。

  本课演示一个"模拟盘 dry-run":把历史数据当作逐日到来的实时行情,
  跑一个带【交易日志】和【风控熔断】的实时循环——这就是实盘代码的骨架。

  实验①  模拟盘资产曲线 + 买卖点
  实验②  交易日志(像券商对账单)
  实验③  风控熔断:回撤超阈值自动清仓停机

运行：python paper_trading_loop.py
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

MAX_DD_STOP = 0.20      # 风控熔断:回撤超过20%就清仓停机
COST = 5/1e4


def paper_trade(close, fast=20, slow=60):
    """模拟盘实时循环:逐日推进,只用当日及以前数据决策,带交易日志与熔断。"""
    cash, shares = 1.0, 0.0
    cur_target = 0.0                 # 当前目标仓位(0/1),只在信号变化时才交易
    equity, log, halted = [], [], False
    peak = 1.0
    for t in range(len(close)):
        price = close.iloc[t]
        cur = cash + shares * price
        peak = max(peak, cur)
        # —— 风控熔断检查(实盘必备的'kill switch') ——
        if not halted and cur < peak * (1 - MAX_DD_STOP):
            if shares > 0:
                cash += shares * price * (1 - COST)
                log.append((close.index[t].date(), "STOP-清仓", price)); shares = 0; cur_target = 0.0
            halted = True
        # —— 策略决策(只看到当日及以前→无前视) ——
        target = 0.0 if halted else _signal(close.iloc[:t+1], fast, slow)
        # —— 只在目标仓位【发生变化】时才下单(避免每日微再平衡) ——
        if target != cur_target:
            cur = cash + shares * price
            desired = target * cur
            trade = desired - shares * price
            cash -= trade + abs(trade) * COST
            shares = desired / price
            cur_target = target
            log.append((close.index[t].date(), "买入" if trade > 0 else "卖出", price))
        equity.append(cash + shares * price)
    return pd.Series(equity, index=close.index), log, halted


def _signal(hist, fast, slow):
    if len(hist) < slow:
        return 0.0
    return 1.0 if hist.iloc[-fast:].mean() > hist.iloc[-slow:].mean() else 0.0


def main():
    print("=" * 60)
    print("  A+4 · 从回测到模拟盘")
    print("=" * 60)
    print("  回测=实验室; 模拟盘=带妆彩排(实时数据+真实规则,假钱); 实盘=正式演出。")
    print("  下面把历史当作'逐日到来的实时行情',跑一个带日志和熔断的实时循环。\n")

    close = qt.load_sample_data()["Close"]
    eq, log, halted = paper_trade(close)

    print("② 交易日志(节选,像券商对账单):")
    for d, action, price in log[:8]:
        print(f"   {d}  {action:<8} @ {price:.2f}")
    print(f"   …… 共 {len(log)} 笔")
    final = eq.iloc[-1]
    mdd = qt.max_drawdown(eq)
    print(f"\n③ 模拟账户结果: 期末净值 {final:.3f} ({final-1:+.1%}), 最大回撤 {mdd:.1%}")
    print(f"   风控熔断: {'触发过(回撤超20%清仓停机)' if halted else '未触发'}\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 资产曲线 + 买卖点
    ax = axes[0, 0]
    ax.plot(eq.index, eq, color="#1f77b4", label="paper account equity")
    buys = [d for d, a, p in log if a == "买入"]
    sells = [d for d, a, p in log if a == "卖出"]
    for d in buys: ax.axvline(pd.Timestamp(d), color="#d62728", alpha=0.15, lw=0.6)
    for d in sells: ax.axvline(pd.Timestamp(d), color="#2ca02c", alpha=0.15, lw=0.6)
    ax.set_title("(1) Paper account equity (red=buy, green=sell lines)")
    ax.set_ylabel("net value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 回撤 + 熔断线
    ax = axes[0, 1]
    dd = eq / eq.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    ax.axhline(-MAX_DD_STOP, color="k", ls="--", label=f"kill-switch -{MAX_DD_STOP:.0%}")
    ax.set_title("(2) Drawdown vs kill-switch line")
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图③ 价格 + 买卖标记
    ax = axes[1, 0]
    ax.plot(close.index, close, color="#333", lw=0.7)
    for d in buys: ax.scatter(pd.Timestamp(d), close.loc[pd.Timestamp(d)], marker="^", color="#d62728", s=30, zorder=5)
    for d in sells: ax.scatter(pd.Timestamp(d), close.loc[pd.Timestamp(d)], marker="v", color="#2ca02c", s=30, zorder=5)
    ax.set_title("(3) Trades on price (^=buy, v=sell)")
    ax.set_ylabel("price"); ax.grid(alpha=0.3)

    # 图④ 上线前检查清单(文字)
    ax = axes[1, 1]; ax.axis("off")
    txt = ("GO-LIVE CHECKLIST\n"
           "  [ ] live code has NO look-ahead\n"
           "  [ ] data feed reliable (gaps? delay?)\n"
           "  [ ] orders confirmed & reconciled\n"
           "  [ ] position/cash matches broker\n"
           "  [ ] risk limits + kill-switch ON\n"
           "  [ ] logging & monitoring\n"
           "  [ ] start TINY, scale slowly\n\n"
           "THE GAUNTLET\n"
           "  backtest -> paper(模拟盘) -> small live -> scale\n"
           "  most strategies die before 'scale'.")
    ax.text(0.02, 0.98, txt, fontsize=9, va="top", family="monospace")
    ax.set_title("Before risking real money")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "A+4配图.png"
    fig.savefig(out, dpi=110)
    print("① 模拟盘=实盘代码的彩排:实时循环、交易日志、风控熔断,缺一不可。")
    print("  图已保存: " + out.name + "（资产/回撤熔断/买卖点/上线清单 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
