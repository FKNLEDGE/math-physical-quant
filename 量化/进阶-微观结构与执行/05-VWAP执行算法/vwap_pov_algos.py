"""
进阶·微观结构与执行 ⑤ VWAP / TWAP / POV 执行算法实务
========================================================
[① Almgren-Chriss](../01-最优执行AlmgrenChriss/) 给了"最优执行"的【理论】(快慢的均值-方差)。
这一课讲交易员真正用的【落地算法】——它们不解微分方程,而是用简单规则把大单拆开:

  · TWAP(时间加权): 每段时间均匀地交易(= AC 的风险中性 λ=0 直线)
  · VWAP(成交量加权): 按【预期成交量曲线】分配(成交量大时多交易)——目标是打平当日 VWAP 基准
  · POV(成交量百分比): 盯着【已实现成交量】,每段只做市场量的固定百分比(自适应)

  实验①  日内成交量是 U 形(开盘/收盘大、午盘小);VWAP 排程跟着它走,TWAP 是平的
  实验②  累计执行曲线:TWAP 直线,VWAP 跟着累计成交量走 S 形
  实验③  跟踪 VWAP 基准的误差:TWAP(大) > VWAP(中) > POV(小);成交量加权才打得平基准
  实验④  没有免费午餐:参与率越高→冲击成本越大(闭环 ① 的快慢权衡)

运行：python vwap_pov_algos.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(0)
N = 39                                    # 一天约 39 个半小时桶
t = np.arange(N)
mid = (N - 1) / 2
PROFILE = 1 + 1.6 * ((t - mid) / mid) ** 2  # U 形日内成交量曲线
PROFILE /= PROFILE.sum()


def sim_day(seed):
    """模拟一天:带噪的已实现成交量(U形)+ 随机游走价格。返回三种算法对 VWAP 基准的跟踪误差(bp)。"""
    r = np.random.default_rng(seed)
    realized = PROFILE * np.exp(0.35 * r.standard_normal(N)); realized /= realized.sum()
    price = 100 + np.cumsum(0.05 * r.standard_normal(N))
    bench = np.sum(price * realized) / np.sum(realized)        # VWAP 基准(已实现量加权)
    twap = np.full(N, 1 / N)                                   # 均匀
    vwap = PROFILE.copy()                                      # 按【预期】曲线
    pov = np.empty(N); pov[0] = PROFILE[0]; pov[1:] = realized[:-1]  # 盯【已实现】(滞后一桶)
    pov /= pov.sum()
    ach = lambda x: np.sum(price * x) / np.sum(x)
    return [(ach(s) - bench) * 1e4 for s in (twap, vwap, pov)], realized, price, (twap, vwap, pov)


def main():
    print("=" * 60)
    print("  进阶·微观结构与执行 ⑤ VWAP / TWAP / POV 执行算法")
    print("=" * 60)
    # 多日统计跟踪误差
    errs = np.array([sim_day(s)[0] for s in range(2000)])
    stds = errs.std(axis=0)
    names = ["TWAP", "VWAP", "POV"]
    print("① 日内成交量 U 形;VWAP 按预期曲线分配,TWAP 均匀,POV 盯已实现量")
    print("③ 对 VWAP 基准的跟踪误差(2000 天, 标准差 bp):")
    for nm, s in zip(names, stds):
        print(f"   {nm:5s} {s:6.1f} bp")
    print(f"   排序 TWAP({stds[0]:.0f}) > VWAP({stds[1]:.0f}) > POV({stds[2]:.0f})：成交量加权才打得平基准。")
    print("   ⚠️ 但 POV 盯已实现量=把'交易多少/何时完成'交给市场(成交量风险):清淡日会做不完。\n")

    # 取一天画排程
    _, realized, price, (twap, vwap, pov) = sim_day(7)

    # ④ 参与率→冲击(闭环①)
    part = np.linspace(0.02, 0.5, 25)                          # 占市场量的百分比
    impact_bps = 30 * np.sqrt(part)                            # 平方根冲击律(示意)

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.bar(t, realized, color="#bbbbbb", label="realized volume (U-shape)")
    ax.plot(t, vwap, "o-", color="#1f77b4", ms=3, label="VWAP schedule (follows volume)")
    ax.plot(t, twap, "s-", color="#d62728", ms=3, label="TWAP schedule (flat)")
    ax.set_title("(1) Intraday volume is U-shaped; VWAP follows it, TWAP is flat")
    ax.set_xlabel("intraday bucket"); ax.set_ylabel("fraction of order / volume")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, np.cumsum(twap), color="#d62728", label="TWAP (straight line)")
    ax.plot(t, np.cumsum(vwap), color="#1f77b4", label="VWAP (S-curve, tracks cum. volume)")
    ax.plot(t, np.cumsum(realized), color="#bbbbbb", ls="--", label="cumulative market volume")
    ax.set_title("(2) Cumulative execution over the day")
    ax.set_xlabel("intraday bucket"); ax.set_ylabel("cumulative fraction done")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bars = ax.bar(names, stds, color=["#d62728", "#1f77b4", "#2ca02c"])
    ax.set_title("(3) Tracking error to VWAP benchmark: volume-weighting wins")
    ax.set_ylabel("tracking-error std (bp, 2000 days)"); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, stds):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=10)

    ax = axes[1, 1]
    ax.plot(part * 100, impact_bps, "-", color="#9467bd", lw=2)
    ax.fill_between(part * 100, impact_bps, alpha=0.15, color="#9467bd")
    ax.set_title("(4) No free lunch: faster (higher participation) → more impact")
    ax.set_xlabel("participation rate (% of market volume)"); ax.set_ylabel("impact cost (bp)")
    ax.annotate("POV lets you pick this rate:\nslow=cheap but volume risk,\nfast=done but costly",
                xy=(35, impact_bps[18]), xytext=(8, impact_bps[-1] * 0.78), fontsize=8.5,
                color="#9467bd", arrowprops=dict(arrowstyle="->", color="#9467bd"))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "VWAP执行配图.png"
    fig.savefig(out, dpi=110)
    print(f"② 累计执行: TWAP 直线 vs VWAP 跟累计成交量的 S 形")
    print(f"④ 参与率→冲击: 平方根律,越快越贵——这正是 ① Almgren-Chriss 的快慢权衡在算法里的样子。\n")
    print(f"  图已保存: {out.name}（成交量曲线/累计执行/跟踪误差/冲击 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
