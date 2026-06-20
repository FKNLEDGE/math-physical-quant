"""
E2 · 反常扩散、Lévy 与分形
==========================
既然收益不是高斯(E1)，价格的『扩散』也就不是正常扩散。物理给了更好的模型：
Lévy 飞行(带大跳跃)、反常扩散(方差∝t^α)、分形自相似(Hurst 指数)。

  实验①  布朗运动(均匀抖动) vs Lévy飞行(成簇+偶发大跳)——两种'游走'
  实验②  反常扩散：均方位移 MSD∝t^(2H)，正常/超/亚扩散三种斜率
  实验③  Hurst 指数 H：H=0.5随机游走，H>0.5趋势持续，H<0.5均值回归
  实验④  分形自相似：把一段路径放大，看起来和整体一样毛糙

运行：python anomalous_diffusion.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(1)


def fbm_chol(n, H, n_paths=1):
    """用 Cholesky 生成分形布朗运动(fBm)，Hurst=H。返回 (n_paths, n+1)。"""
    k = np.arange(n)
    g = 0.5*(np.abs(k+1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k-1)**(2*H))  # fGn 自协方差
    cov = np.array([[g[abs(i-j)] for j in range(n)] for i in range(n)])
    L = np.linalg.cholesky(cov + 1e-10*np.eye(n))
    fgn = (L @ RNG.standard_normal((n, n_paths))).T
    return np.cumsum(np.concatenate([np.zeros((n_paths, 1)), fgn], axis=1), axis=1)


def main():
    print("=" * 60)
    print("  E2 · 反常扩散、Lévy 与分形")
    print("=" * 60)
    print("  正常扩散(布朗,B5)：方差∝t，标准差∝√t，高斯增量。")
    print("  反常扩散：方差∝t^(2H)。H=0.5正常；H>0.5超扩散(趋势)；H<0.5亚扩散(回归)。\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 布朗 vs Lévy 飞行 (2D)
    nstep = 1500
    # 布朗：高斯步长
    bw = np.cumsum(RNG.standard_normal((nstep, 2)), axis=0)
    # Lévy 飞行：方向均匀，步长幂律(Pareto, 偶发巨跳)
    ang = RNG.uniform(0, 2*np.pi, nstep)
    mag = (1 - RNG.uniform(0, 1, nstep))**(-1/1.5)
    steps = np.c_[mag*np.cos(ang), mag*np.sin(ang)]
    lf = np.cumsum(steps, axis=0)
    ax = axes[0, 0]
    ax.plot(bw[:, 0], bw[:, 1], lw=0.6, color="#1f77b4")
    ax.set_title("(1a) Brownian motion: uniform jitter")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(lf[:, 0], lf[:, 1], lw=0.6, color="#d62728")
    ax.set_title("(1b) Levy flight: clusters + rare huge jumps")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3)
    print("① 布朗运动:处处均匀的小抖动(高斯)。Lévy飞行:大部分小步,偶尔一个巨跳——")
    print("   后者更像真实价格(平静中突然跳空)。Lévy步长无限方差,中心极限定理失效。\n")

    # 图② 反常扩散 MSD ∝ t^(2H)
    ax = axes[1, 0]
    n, M = 500, 400
    ts = np.arange(1, n+1)
    for H, c, name in [(0.7, "#d62728", "H=0.7 super (trend)"),
                       (0.5, "#1f77b4", "H=0.5 normal"),
                       (0.3, "#2ca02c", "H=0.3 sub (revert)")]:
        paths = fbm_chol(n, H, M)
        msd = (paths[:, 1:]**2).mean(axis=0)
        ax.loglog(ts, msd, color=c, label=name)
    ax.set_title("(2) Anomalous diffusion: MSD ~ t^(2H)")
    ax.set_xlabel("time (log)"); ax.set_ylabel("mean squared displacement (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    print("② 均方位移 MSD∝t^(2H)：双对数下斜率=2H。超扩散(H>.5)散得比正常快(趋势),")
    print("   亚扩散(H<.5)慢(回归)。布朗(H=.5)斜率=1，正是√t律的平方。\n")

    # 图③ Hurst 指数估计
    ax = axes[1, 1]
    Hs_true = [0.7, 0.5, 0.3]
    Hs_est = []
    lags = np.arange(2, 80)
    for H in Hs_true:
        path = fbm_chol(2000, H, 1)[0]
        # 用增量的标准差随 lag 的标度估 H： std(X[t+τ]-X[t]) ∝ τ^H
        rs = [np.std(path[lag:] - path[:-lag]) for lag in lags]
        Hh = np.polyfit(np.log(lags), np.log(rs), 1)[0]
        Hs_est.append(Hh)
    x = np.arange(3)
    ax.bar(x-0.2, Hs_true, 0.4, color="#888", label="true H")
    ax.bar(x+0.2, Hs_est, 0.4, color="#ff7f0e", label="estimated H")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["persistent", "random walk", "mean-revert"])
    ax.set_title("(3) Hurst exponent recovers the memory")
    ax.set_ylabel("H"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    print("③ Hurst 指数：从增量标度 std(ΔX over τ)∝τ^H 估计。")
    for n_, t_, e_ in zip(["趋势持续", "随机游走", "均值回归"], Hs_true, Hs_est):
        print(f"   {n_}: 真H={t_} 估H={e_:.2f}")
    print("   H 一根旋钮统一了趋势/随机/回归。真实股市H≈0.5(略有结构),波动率序列H>0.5(长记忆)。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "E2配图.png"
    fig.savefig(out, dpi=110)
    # 文本：自相似
    print("④ 分形自相似：fBm 路径在任何尺度放大,统计上一样毛糙(自仿射)。")
    print("   Mandelbrot:价格图去掉坐标轴,你分不清是1分钟还是1天的——市场是分形的。")
    print(f"\n  图已保存: {out.name}（布朗/Lévy/反常扩散/Hurst 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
