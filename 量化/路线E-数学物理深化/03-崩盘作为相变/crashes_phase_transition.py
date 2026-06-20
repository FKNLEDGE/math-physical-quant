"""
E3 · 崩盘作为相变/临界现象
==========================
E1/E2 的厚尾、幂律、长记忆从哪来？答案:【相互作用 + 临界】。
当交易者足够强地相互模仿(羊群)，市场会像水结冰一样发生『相变』，
崩盘就是那个『临界点』——这里涌现出幂律涨落。

  实验①  Ising 市场的相变：序参量(市场情绪) vs 相互作用强度
  实验②  临界涨落：易感性(涨落幅度)在临界点暴涨——这就是大波动/崩盘的来源
  实验③  临界点的快照：买卖簇呈现各种尺度(分形)——幂律的微观图像
  实验④  Sornette 对数周期幂律(LPPL)：泡沫超指数加速 + 加速振荡 → 崩盘

运行：python crashes_phase_transition.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(0)
L = 48  # 交易者格点 L×L


def ising_run(T, n_eq=200, n_meas=200):
    """2D Ising(向量化棋盘 Metropolis)。spin=买(+1)/卖(-1)，T=非理性/独立程度。"""
    s = np.ones((L, L), dtype=int)        # 从有序态出发，避免低温退火残留畴壁(让序参量曲线干净)
    i, j = np.indices((L, L))
    mask = (i + j) % 2
    ms = []
    snap = None
    for step in range(n_eq + n_meas):
        for color in [0, 1]:
            nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1))
            dE = 2 * s * nb
            flip = (mask == color) & ((dE < 0) | (RNG.random((L, L)) < np.exp(-dE / T)))
            s[flip] *= -1
        if step >= n_eq:
            ms.append(s.mean())
        if step == n_eq + n_meas - 1:
            snap = s.copy()
    ms = np.array(ms)
    chi = L * L * ms.var() / T              # 易感性(涨落)
    return np.abs(ms).mean(), chi, snap


def main():
    print("=" * 60)
    print("  E3 · 崩盘作为相变/临界现象")
    print("=" * 60)
    print("  Ising 市场：每个交易者是一个自旋 买(+1)/卖(-1)，受邻居影响(模仿/羊群)。")
    print("  T 大=各自独立(无序,小波动)；T 小=强烈模仿(有序,羊群);临界 Tc≈2.27 处涌现幂律。\n")

    Ts = np.linspace(1.4, 3.4, 22)
    mags, chis = [], []
    snaps = {}
    for T in Ts:
        m, chi, snap = ising_run(T)
        mags.append(m); chis.append(chi)
        if abs(T-1.4) < 0.06: snaps["order"] = snap
        if abs(T-2.27) < 0.06: snaps["crit"] = snap
        if abs(T-3.4) < 0.06: snaps["disorder"] = snap
    mags, chis = np.array(mags), np.array(chis)
    Tc = Ts[np.argmax(chis)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 序参量(市场情绪) vs T
    ax = axes[0, 0]
    ax.plot(Ts, mags, "o-", color="#1f77b4")
    ax.axvline(Tc, color="red", ls="--", label=f"critical Tc≈{Tc:.2f}")
    ax.set_title("(1) Phase transition: market consensus vs interaction")
    ax.set_xlabel("T (less interaction ->)"); ax.set_ylabel("order parameter |sentiment|")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"① 序参量(羊群程度)：T小(强模仿)→接近1(全体一致,有序);T大→0(各自独立,无序)。")
    print(f"   在 Tc≈{Tc:.2f} 处发生突变=相变。\n")

    # 图② 易感性(涨落) vs T —— 临界点暴涨
    ax = axes[0, 1]
    ax.plot(Ts, chis, "o-", color="#d62728")
    ax.axvline(Tc, color="red", ls="--", label=f"Tc≈{Tc:.2f}")
    ax.set_title("(2) Susceptibility PEAKS at criticality = big swings")
    ax.set_xlabel("T"); ax.set_ylabel("fluctuation (susceptibility)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("② 易感性(市场对微小扰动的反应/涨落幅度)在临界点【暴涨】——")
    print("   一点点风吹草动就能引发巨幅波动。这正是大涨大跌/崩盘的物理来源。\n")

    # 图③ 临界点快照：各种尺度的簇(分形)
    ax = axes[1, 0]
    ax.imshow(snaps.get("crit", snaps["disorder"]), cmap="coolwarm", interpolation="nearest")
    ax.set_title("(3) At criticality: buy/sell clusters of ALL sizes (fractal)")
    ax.set_xticks([]); ax.set_yticks([])
    print("③ 临界点快照：买卖簇出现【各种尺度】(大簇套小簇)=分形/无标度→幂律(闭环E1)。")
    print("   远离临界:要么一片同色(有序),要么细碎噪点(无序),都没有跨尺度结构。\n")

    # 图④ Sornette LPPL：泡沫 → 崩盘
    ax = axes[1, 1]
    tc = 1.0
    t = np.linspace(0, 0.985, 1000)
    beta, omega, phi = 0.34, 9.0, 0.0
    # LPPL: price = A - B(tc-t)^β[1 + C cos(ω log(tc-t) + φ)]，超指数加速 + 对数周期振荡
    A, B, C = 220, 80, 0.12
    price = A - B*(tc - t)**beta * (1 + C*np.cos(omega*np.log(tc - t) + phi))
    ax.plot(t, price, color="#9467bd")
    ax.axvline(tc, color="red", ls="--", label="critical time tc (crash)")
    ax.set_title("(4) Sornette LPPL: bubble accelerates to a critical time")
    ax.set_xlabel("time"); ax.set_ylabel("price")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("④ Sornette 对数周期幂律(LPPL)：泡沫不是匀速涨,而是【超指数】加速,")
    print("   并伴随【越来越密】的振荡(对数周期),逼近一个临界时间 tc——崩盘点。")
    print("   它把崩盘当相变来建模,给出(有争议的)预警。1987/2000/2008 事后拟合得不错。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "E3配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（相变/临界涨落/分形快照/LPPL 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
