"""
A+1 · 仓位管理与资金管理（Kelly 与破产风险）
============================================
有了优势(edge)还不够——【下多大注】决定你是富有还是破产。
这节课用模拟告诉你:同样的优势,押注大小不同,命运天差地别。

  实验①  同样+10%优势,4个赌注大小的财富路径:满仓必破产,Kelly稳增长
  实验②  Kelly曲线:增长率随押注比例先升后降,在f*达到最优,过2f*变负
  实验③  破产风险随押注变大急剧上升
  实验④  全Kelly vs 半Kelly:半Kelly拿走大部分增长,回撤却小得多

运行：python position_sizing.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(0)
P, B = 0.55, 1.0                 # 赢的概率55%,赔率1(押1赢1,即+10%优势的偏向硬币)
F_STAR = (P*(B+1) - 1)/B         # Kelly 最优比例 = 2p-1 = 0.10


def growth_rate(f):
    """长期对数增长率 g(f)=p·ln(1+fb)+(1-p)·ln(1-f)。"""
    return P*np.log(1 + f*B) + (1-P)*np.log(1 - f)


def simulate(f, n_bets=250, n_paths=2000):
    """模拟下注比例 f 的财富路径(每次按比例押注偏向硬币)。"""
    wins = RNG.random((n_paths, n_bets)) < P
    mult = np.where(wins, 1 + f*B, 1 - f)
    return np.cumprod(np.concatenate([np.ones((n_paths, 1)), mult], axis=1), axis=1)


def main():
    print("=" * 60)
    print("  A+1 · 仓位管理与资金管理（Kelly 与破产风险）")
    print("=" * 60)
    print(f"  游戏:赢概率 p={P}, 赔率 b={B}(押1赢1)。这是个有正优势的游戏。")
    print(f"  Kelly 最优押注比例 f* = 2p-1 = {F_STAR:.0%}（每次押总资金的 {F_STAR:.0%}）\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 四种押注的财富路径(取中位数路径示意)
    ax = axes[0, 0]
    for f, c, name in [(0.05, "#2ca02c", "5% (half-Kelly)"), (0.10, "#1f77b4", "10% (Kelly f*)"),
                       (0.30, "#ff7f0e", "30% (over-bet)"), (1.0, "#d62728", "100% (all-in)")]:
        paths = simulate(f)
        med = np.median(paths, axis=0)
        ax.plot(med, color=c, label=name)
    ax.set_yscale("log")
    ax.set_title("(1) Same edge, different bet size -> different fate")
    ax.set_xlabel("number of bets"); ax.set_ylabel("wealth (log, median)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("① 同样+10%优势,押注比例不同:")
    print("   满仓(100%):一次必输的硬币就归零→破产;过度(30%):缓慢下行;")
    print(f"   Kelly(10%):稳定增长最快;半Kelly(5%):稍慢但更稳。【优势相同,命运由仓位决定】\n")

    # 图② Kelly 曲线
    ax = axes[0, 1]
    fs = np.linspace(0, 0.5, 200)
    g = growth_rate(fs)
    ax.plot(fs, g, color="#1f77b4")
    ax.axvline(F_STAR, color="green", ls="--", label=f"Kelly f*={F_STAR:.2f}")
    ax.axhline(0, color="k", lw=0.6)
    f_zero = 2*F_STAR
    ax.axvline(f_zero, color="red", ls="--", label=f"zero-growth ~{f_zero:.2f}")
    ax.set_title("(2) Kelly curve: growth peaks at f*, negative past 2f*")
    ax.set_xlabel("bet fraction f"); ax.set_ylabel("long-run growth rate g(f)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"② Kelly曲线:增长率在 f*={F_STAR:.0%} 最高;超过约 2f*={f_zero:.0%} 后增长变【负】→")
    print("   押太大,即使有优势也长期亏损(过度下注的诅咒)。下注是个'过犹不及'的凸性问题。\n")

    # 图③ 破产风险 vs 押注
    ax = axes[1, 0]
    fs2 = np.linspace(0.02, 0.6, 30)
    ruin = []
    for f in fs2:
        paths = simulate(f, n_bets=250, n_paths=3000)
        ruin.append((paths[:, -1] < 0.5).mean())   # 财富跌破初始一半算"重伤"
    ax.plot(fs2, ruin, "o-", color="#d62728")
    ax.axvline(F_STAR, color="green", ls="--", label=f"Kelly f*={F_STAR:.2f}")
    ax.set_title("(3) Risk of ruin rises sharply with bet size")
    ax.set_xlabel("bet fraction f"); ax.set_ylabel("P(wealth < half of start)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("③ 破产/重伤风险随押注变大急剧上升。Kelly附近已不低,所以实务常用更保守的押注。\n")

    # 图④ 全Kelly vs 半Kelly
    ax = axes[1, 1]
    full = simulate(F_STAR, n_bets=250, n_paths=3000)
    half = simulate(F_STAR/2, n_bets=250, n_paths=3000)
    def max_dd(paths):
        peak = np.maximum.accumulate(paths, axis=1)
        return ((paths/peak) - 1).min(axis=1)
    ax.hist(max_dd(full), bins=40, alpha=0.6, color="#1f77b4", label="full Kelly")
    ax.hist(max_dd(half), bins=40, alpha=0.6, color="#2ca02c", label="half Kelly")
    ax.set_title("(4) Half-Kelly: most of the growth, far less drawdown")
    ax.set_xlabel("max drawdown"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    gf = np.median(full[:, -1]); gh = np.median(half[:, -1])
    print(f"④ 全Kelly vs 半Kelly(250注,中位终值): 全={gf:.1f}x, 半={gh:.1f}x")
    print(f"   半Kelly增长约为全Kelly的 {np.log(gh)/np.log(gf):.0%},但最大回撤显著更小→")
    print("   实务里几乎没人用全Kelly,多用 1/4~1/2 Kelly,牺牲一点增长换大幅降低回撤。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "A+1配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（财富路径/Kelly曲线/破产风险/全vs半Kelly 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
