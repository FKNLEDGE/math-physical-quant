"""
第 05 课 · 给策略祛魅（量化主线高潮）
=====================================
第 04 课的策略跑输了基准。但更危险的是那些『看起来很赚』的策略——
它们往往死于下面四个陷阱。这节课用四个实验，教你像科学家一样拷问自己：

  刀①  前视偏差：偷看未来能把垃圾策略吹成神话
  刀②  过拟合  ：在历史上挑到的『最优参数』，换段时间就失灵
  刀③  交易成本：换手越勤，成本越能吃光利润
  刀④  信号vs噪声：同一策略在 200 个平行世界里，多少是运气？

运行：python lesson05_reality_check.py
"""

import sys
import importlib.util
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quant_tools as qt


def load_generator():
    """加载数据生成器（用于刀④的『平行世界』蒙特卡洛）。"""
    path = Path(__file__).resolve().parents[1] / "数据" / "生成示例数据.py"
    spec = importlib.util.spec_from_file_location("gen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def knife1_lookahead(close):
    print("\n刀① 前视偏差：偷看未来 = 自欺欺人")
    mkt = qt.to_log_returns(close)

    # 极端版：先知，只在上涨日持仓（用了当天/未来信息）
    perfect = mkt.clip(lower=0)
    oracle_total = np.exp(perfect.sum()) - 1
    print(f"   『先知』(只吃上涨日)总收益: {oracle_total:,.0%}  vs 买入持有 {np.exp(mkt.sum())-1:.0%}")
    print("   —— 荒谬地高，这就是泄漏未来信息的威力。")

    # 隐蔽版：用【全样本】均值标准化（偷看了含未来的整段），vs 只用过去的滚动标准化
    z_full = (close - close.mean()) / close.std()                                  # 泄漏！
    z_roll = (close - close.rolling(120).mean()) / close.rolling(120).std()         # 诚实
    eq = {}
    for name, z in [("全样本z(泄漏)", z_full), ("滚动z(诚实)", z_roll)]:
        pos = (z < 0).astype(float).shift(1).fillna(0)   # z<0 视为便宜→买
        bt = qt.backtest_long_only(close, pos, cost_bps=5)
        eq[name] = bt["策略净值"]
        print(f"   {name}: 总收益 {qt.perf_metrics(bt['策略收益'])['总收益']:7.1%}")
    print("   —— 同一个想法，仅仅因为标准化时偷看了未来，就从亏损变成『大赚』。")
    return eq


def knife2_overfitting(close, fasts, slows):
    print("\n刀② 过拟合：历史最优参数，换段时间就翻车")
    split = int(len(close) * 0.67)
    ins, outs = close.iloc[:split], close.iloc[split:]

    # 网格搜索：在样本内找夏普最高的参数
    grid = np.full((len(fasts), len(slows)), np.nan)
    best = None
    for i, f in enumerate(fasts):
        for j, s in enumerate(slows):
            if f >= s:
                continue
            m = qt.perf_metrics(qt.backtest_long_only(ins, qt.ma_crossover_position(ins, f, s), 5)["策略收益"])
            grid[i, j] = m["夏普比率"]
            if best is None or m["夏普比率"] > best[0]:
                best = (m["夏普比率"], f, s)
    is_sharpe, bf, bs = best
    oos = qt.perf_metrics(qt.backtest_long_only(outs, qt.ma_crossover_position(outs, bf, bs), 5)["策略收益"])
    print(f"   样本内最优参数 = {bf}/{bs}")
    print(f"   样本内夏普 {is_sharpe:5.2f}  ->  样本外夏普 {oos['夏普比率']:5.2f}")
    print("   —— 试了一堆参数挑最好的那组，多半是『拟合了历史的噪声』，对未来没用。")
    return grid, (bf, bs), split


def knife3_costs(close):
    print("\n刀③ 交易成本：换手越勤，越被成本吃掉")
    for f, s in [(5, 20), (20, 60)]:
        pos = qt.ma_crossover_position(close, f, s)
        turns = int(pos.diff().abs().fillna(0).gt(0).sum())
        gross = qt.perf_metrics(qt.backtest_long_only(close, pos, 0)["策略收益"])["总收益"]
        net = qt.perf_metrics(qt.backtest_long_only(close, pos, 20)["策略收益"])["总收益"]
        print(f"   均线{f:>2}/{s:<3}(换手{turns:>3}次): 毛收益 {gross:7.1%} -> 扣20bp {net:7.1%}  (损耗 {gross-net:5.1%})")
    print("   —— 一个朴素检验：把成本加倍策略还活吗？活不下来的多是假策略。")


def knife4_montecarlo(gen, n_worlds=200):
    print(f"\n刀④ 信号vs噪声：同一策略跑 {n_worlds} 个平行世界")
    diffs = []
    for sd in range(n_worlds):
        c = pd.Series(gen.generate_close(6 * 252, sd))
        bh = np.exp(qt.to_log_returns(c).sum()) - 1
        st = qt.perf_metrics(qt.backtest_long_only(c, qt.ma_crossover_position(c, 20, 60), 5)["策略收益"])["总收益"]
        diffs.append(st - bh)
    diffs = np.array(diffs)
    win = (diffs > 0).mean()
    print(f"   均线跑赢买入持有的比例: {win:.0%}    平均超额: {diffs.mean():+.1%}")
    print(f"   —— 哪怕这个『带趋势』的世界对趋势策略有利，仍有 {1-win:.0%} 的概率跑输。")
    print("      真实市场趋势更弱、更善变。单次漂亮回测，极可能只是运气。")
    return diffs


def main():
    print("=" * 62)
    print("  第 05 课 · 给策略祛魅：四把刀拷问『真信号还是运气』")
    print("=" * 62)

    df = qt.load_sample_data()
    close = df["Close"]
    gen = load_generator()

    fasts = [5, 10, 15, 20, 30, 40]
    slows = [50, 75, 100, 150, 200]

    eq_leak = knife1_lookahead(close)
    grid, (bf, bs), split = knife2_overfitting(close, fasts, slows)
    knife3_costs(close)
    diffs = knife4_montecarlo(gen)

    # ===== 画一个 2×2 的『现实拷问仪表盘』=====
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) 前视偏差：泄漏 vs 诚实 的净值
    ax = axes[0, 0]
    ax.plot(eq_leak["全样本z(泄漏)"].index, eq_leak["全样本z(泄漏)"], color="#d62728", label="full-sample z (LEAK)")
    ax.plot(eq_leak["滚动z(诚实)"].index, eq_leak["滚动z(诚实)"], color="#1f77b4", label="rolling z (honest)")
    ax.set_title("Knife 1: look-ahead inflates a losing idea")
    ax.set_ylabel("net value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) 过拟合：参数热力图（样本内夏普）
    ax = axes[0, 1]
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(slows)))
    ax.set_xticklabels(slows)
    ax.set_yticks(range(len(fasts)))
    ax.set_yticklabels(fasts)
    ax.set_xlabel("slow MA")
    ax.set_ylabel("fast MA")
    ax.set_title("Knife 2: in-sample Sharpe by params\n(cherry-picking the best = overfitting)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (1,0) 过拟合：最优参数的 样本内 vs 样本外 净值
    ax = axes[1, 0]
    bt_best = qt.backtest_long_only(close, qt.ma_crossover_position(close, bf, bs), 5)
    eqb = bt_best["策略净值"]
    split_date = close.index[split]
    ax.plot(eqb.index, eqb, color="#9467bd")
    ax.axvline(split_date, color="k", ls="--", lw=1)
    ax.text(eqb.index[int(split*0.35)], eqb.max()*0.98, "in-sample\n(optimized here)", fontsize=8, va="top")
    ax.text(split_date, eqb.min()*1.02, " out-of-sample →", fontsize=8)
    ax.set_title(f"Knife 2: best params {bf}/{bs} — great in, weak out")
    ax.set_ylabel("net value")
    ax.grid(alpha=0.3)

    # (1,1) 蒙特卡洛：超额收益分布
    ax = axes[1, 1]
    ax.hist(diffs, bins=30, color="#ff7f0e", edgecolor="white")
    ax.axvline(0, color="k", lw=1.2)
    ax.axvline(diffs.mean(), color="red", ls="--", lw=1, label=f"mean {diffs.mean():+.1%}")
    ax.set_title(f"Knife 4: strategy − buy&hold over 200 worlds\n(win {((diffs>0).mean()):.0%}, the rest is luck)")
    ax.set_xlabel("excess total return")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "第05课配图.png"
    fig.savefig(out, dpi=110)
    print(f"\n  图已保存: {out.name}（前视/过拟合/参数热图/蒙特卡洛 四合一）")

    print("\n" + "=" * 62)
    print("  把这张『诚实清单』钉在脑子里（详见 docs/02-方法论与思维模型）：")
    print("  □ 调了太多参数？ □ 试了太多次挑最好？ □ 偷用未来信息？")
    print("  □ 含已退市标的？ □ 加上成本还赚？ □ 换段历史还成立？ □ 说得清何时算错？")
    print("=" * 62)


if __name__ == "__main__":
    main()
