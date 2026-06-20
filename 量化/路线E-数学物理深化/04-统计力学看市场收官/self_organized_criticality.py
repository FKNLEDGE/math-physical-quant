"""
E4 · 统计力学看市场（自组织临界 + 全项目收官）
=============================================
E3 说『临界点』涌现幂律,但谁把市场调到临界?答案:它【自己】调过去的。
Bak 的沙堆模型:简单局部规则,系统自发演化到临界态,雪崩大小服从幂律——
无需任何精调。这解释了为什么真实市场『天然』充满幂律(E1):自组织临界(SOC)。

  实验①  沙堆的自组织临界态(高度图)
  实验②  雪崩大小分布:幂律(双对数直线)——各种尺度的'崩盘'
  实验③  雪崩时间序列:大量小崩 + 偶发巨崩(像极了市场)
  实验④  涌现的'价格':简单规则→市场般的暴涨暴跌(平静被突变打断)

运行：python self_organized_criticality.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(0)
L = 50


def run_sandpile(n_grains, burn=8000):
    """Bak-Tang-Wiesenfeld 沙堆：随机加沙,满4则坍塌给四邻,边界耗散。"""
    grid = RNG.integers(0, 4, size=(L, L))
    sizes = []
    for step in range(n_grains):
        grid[RNG.integers(L), RNG.integers(L)] += 1
        avalanche = 0
        while np.any(grid >= 4):
            topple = grid >= 4
            avalanche += int(topple.sum())
            grid[topple] -= 4
            grid[1:, :] += topple[:-1, :]      # 向下传
            grid[:-1, :] += topple[1:, :]      # 向上传
            grid[:, 1:] += topple[:, :-1]      # 向右传
            grid[:, :-1] += topple[:, 1:]      # 向左传(边界外的沙子耗散掉)
        if step >= burn:
            sizes.append(avalanche)
    return grid, np.array(sizes)


def main():
    print("=" * 60)
    print("  E4 · 统计力学看市场（自组织临界 + 收官）")
    print("=" * 60)
    print("  E3:临界点涌现幂律。但谁把市场调到临界？SOC:系统【自己】演化到临界。")
    print("  沙堆:不断加沙,堆到临界坡度,此后一粒沙可引发任意大小的雪崩(幂律)。\n")

    grid, sizes = run_sandpile(30000)
    sizes = sizes[sizes > 0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 自组织临界态
    ax = axes[0, 0]
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title("(1) Self-organized critical state (grain heights)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    print("① 沙堆自发演化到一个'临界'高度分布——没人去精调,它自己到了临界。\n")

    # 图② 雪崩大小分布(幂律)
    ax = axes[0, 1]
    bins = np.logspace(0, np.log10(sizes.max()+1), 25)
    hist, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1]*edges[1:])
    good = hist > 0
    ax.loglog(centers[good], hist[good], "o", color="#d62728")
    # 拟合幂律指数
    m = good & (centers > 3) & (centers < sizes.max()/3)
    tau = -np.polyfit(np.log(centers[m]), np.log(hist[m]), 1)[0]
    xf = centers[m]
    ax.loglog(xf, hist[m][0]*(xf/xf[0])**(-tau), "k--", label=f"power law tau~{tau:.2f}")
    ax.set_title("(2) Avalanche sizes: POWER LAW (crashes of all sizes)")
    ax.set_xlabel("avalanche size (log)"); ax.set_ylabel("probability (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    print(f"② 雪崩大小服从幂律(双对数直线,指数τ≈{tau:.2f})——'崩盘'有各种尺度,无典型大小。")
    print("   小崩频繁,大崩罕见但必然,同一条幂律统辖。这是 E1 厚尾的『自组织』来源。\n")

    # 图③ 雪崩时间序列
    ax = axes[1, 0]
    ax.plot(sizes[:2000], color="#1f77b4", lw=0.6)
    ax.set_title("(3) Avalanches over time: calm + sudden big ones")
    ax.set_xlabel("event"); ax.set_ylabel("avalanche size")
    ax.grid(alpha=0.3)
    print(f"③ 雪崩时间序列:绝大多数极小,偶尔一个巨大(最大 {sizes.max()})——")
    print("   像极了市场:长期平静被突发暴跌打断(波动聚集/间歇性)。\n")

    # 图④ 涌现的'价格'
    ax = axes[1, 1]
    signs = RNG.choice([-1, 1], size=len(sizes))
    price = 100 + np.cumsum(signs*np.sqrt(sizes))*0.5    # 把雪崩当价格冲击累加
    ax.plot(price, color="#9467bd", lw=0.8)
    ax.set_title("(4) Emergent 'price': simple rules -> market-like bursts")
    ax.set_xlabel("time"); ax.set_ylabel("price")
    ax.grid(alpha=0.3)
    print("④ 把雪崩当价格冲击累加,涌现出一条'价格'——平静中突发暴涨暴跌,")
    print("   厚尾、聚集、崩盘全自发出现。【简单规则 → 复杂市场】,这就是涌现(emergence)。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "E4配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（临界态/幂律雪崩/雪崩时序/涌现价格 四合一）")
    print("=" * 60)
    print("  🎓 全项目收官：从『量化是什么』到『市场的统计物理』，")
    print("     数学物理与量化在此彻底合流。详见 笔记.md 的总收束。")
    print("=" * 60)


if __name__ == "__main__":
    main()
