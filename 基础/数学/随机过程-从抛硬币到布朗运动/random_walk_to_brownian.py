"""
随机过程 · 从抛硬币到布朗运动（四个实验）
==========================================
一步步把『随机性如何随时间演化』看明白：

  实验①  随机游走：把抛硬币(±1)累加，就是最简单的『股价』
          —— 看到方差随步数线性增长 → 标准差按 √n（第03课 √252 的来历！）
  实验②  从离散到连续：步子越细，随机游走越逼近『布朗运动』
  实验③  伊藤的核心秘密：(ΔW)² ≈ Δt（普通微积分在这失效）
  实验④  几何布朗运动与『波动率拖累』：第01课 simulate_gbm 里 -½σ² 的来历

运行：python random_walk_to_brownian.py
图内文字用英文（避免缺中文字体显示方块）；讲解在终端与 笔记.md。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(7)


def exp1_random_walk(ax_paths, ax_hist):
    """实验①：抛硬币随机游走，方差随步数线性增长，标准差 ~ √n。"""
    n_steps, n_paths = 200, 3000
    steps = RNG.choice([-1, 1], size=(n_paths, n_steps))   # 每步 ±1，各 50%
    walks = np.cumsum(steps, axis=1)                        # 累加 = 随机游走

    # 画 30 条路径 + 理论 ±√n 包络
    t = np.arange(1, n_steps + 1)
    for i in range(30):
        ax_paths.plot(t, walks[i], lw=0.6, alpha=0.5)
    ax_paths.plot(t, np.sqrt(t), "k--", lw=1.5, label="+/- sqrt(n) envelope")
    ax_paths.plot(t, -np.sqrt(t), "k--", lw=1.5)
    ax_paths.set_title("(1) Random walks: spread grows as sqrt(n)")
    ax_paths.set_xlabel("step n")
    ax_paths.set_ylabel("position")
    ax_paths.legend(fontsize=8)
    ax_paths.grid(alpha=0.3)

    # 终点分布 vs 理论正态 N(0, n)
    endpoints = walks[:, -1]
    ax_hist.hist(endpoints, bins=40, density=True, color="#1f77b4",
                 edgecolor="white", alpha=0.8)
    xs = np.linspace(endpoints.min(), endpoints.max(), 200)
    ax_hist.plot(xs, np.exp(-xs**2 / (2*n_steps)) / np.sqrt(2*np.pi*n_steps),
                 "r-", lw=2, label=f"Normal(0, {n_steps})")
    ax_hist.set_title("(2) Endpoint distribution -> Normal")
    ax_hist.set_xlabel(f"position after {n_steps} steps")
    ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    print("① 随机游走（抛硬币 ±1，累加）")
    print(f"   理论：走 {n_steps} 步后，方差 = 步数 = {n_steps}，标准差 = √{n_steps} = {np.sqrt(n_steps):.2f}")
    print(f"   实测：{n_paths} 条路径终点的标准差 = {endpoints.std():.2f}  （吻合 ✅）")
    print(f"   🔑 标准差按 √n 增长 —— 这就是第03课『年化波动 = 日波动 × √252』的根源！\n")


def exp2_brownian_limit(ax):
    """实验②：固定总时间 T，步子越细，随机游走越像连续的布朗运动。"""
    T = 1.0
    fine_N = 4000
    # 先生成一条很细的布朗运动，再以不同粗细『采样』，展示逼近过程
    dt = T / fine_N
    dW = np.sqrt(dt) * RNG.standard_normal(fine_N)
    W = np.concatenate([[0], np.cumsum(dW)])
    t_fine = np.linspace(0, T, fine_N + 1)
    for N, color, lw in [(20, "#2ca02c", 1.6), (200, "#ff7f0e", 1.0), (4000, "#1f77b4", 0.6)]:
        idx = np.linspace(0, fine_N, N + 1).astype(int)
        ax.plot(t_fine[idx], W[idx], color=color, lw=lw, label=f"N={N} steps")
    ax.set_title("(3) Finer steps -> Brownian motion (continuous, jagged)")
    ax.set_xlabel("time")
    ax.set_ylabel("W(t)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("② 从离散到连续：固定总时间，步子从 20 → 200 → 4000")
    print("   步子越细，折线越逼近一条『处处连续、却处处不光滑』的曲线 = 布朗运动")
    print("   它在任何尺度放大看都一样毛糙（自相似/分形，呼应 Mandelbrot）。\n")


def exp3_ito_insight(ax):
    """实验③：伊藤的核心——(ΔW)² 的累加收敛到时间 T，而不是 0。"""
    T, N = 1.0, 2000
    dt = T / N
    dW = np.sqrt(dt) * RNG.standard_normal(N)
    t = np.linspace(dt, T, N)
    sum_dW = np.cumsum(dW)            # ΔW 的累加 = W(t)，到处乱走 ~ N(0,T)
    sum_dW2 = np.cumsum(dW**2)        # (ΔW)² 的累加 → 收敛到 T（二次变差）

    ax.plot(t, sum_dW2, color="#d62728", lw=1.5, label="sum of (dW)^2  ->  T")
    ax.plot(t, t, "k--", lw=1, label="y = t (the limit)")
    ax.plot(t, sum_dW, color="#1f77b4", lw=0.8, alpha=0.7, label="sum of dW = W(t) (wanders)")
    ax.set_title("(4) Ito's secret: (dW)^2 sums to t, not 0")
    ax.set_xlabel("time")
    ax.set_ylabel("cumulative sum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    print("③ 伊藤的核心秘密：(ΔW)² 的累加")
    print(f"   把每小步的 (ΔW)² 加起来 = {sum_dW2[-1]:.3f}  ≈ T = {T}（不是 0！）")
    print("   普通微积分里 (dx)² 是高阶小量、可忽略；但布朗运动 (dW)²≈dt 不可忽略。")
    print("   👉 这一条，正是『随机微积分』区别于普通微积分的根本，也是伊藤引理的来源。\n")


def exp4_volatility_drag():
    """实验④：几何布朗运动与波动率拖累——第01课 -½σ² 的来历（纯文本+数值）。"""
    print("④ 几何布朗运动(GBM) 与『波动率拖累』")
    print("   先看一个直觉：涨 10% 再跌 10%，回到原点了吗？")
    v = 1.0 * 1.10 * 0.90
    print(f"      1 →涨10%→ 1.10 →跌10%→ {v:.4f}   亏了 {1-v:.2%}（≈ ½×0.1² = {0.5*0.1**2:.2%}）")
    print("   波动本身就在吃掉复利，这叫『波动率拖累』。")
    # 用模拟印证：log 价格的漂移 = μ - ½σ²
    mu, sigma, T, N, M = 0.0, 0.40, 1.0, 252, 20000
    dt = T / N
    dW = np.sqrt(dt) * RNG.standard_normal((M, N))
    logS = np.cumsum((mu - 0.5*sigma**2)*dt + sigma*dW, axis=1)
    S_T = np.exp(logS[:, -1])
    print(f"\n   模拟 {M} 条 GBM（μ={mu}, σ={sigma}）：")
    print(f"      理论 E[lnS_T] = (μ-½σ²)T = {(mu-0.5*sigma**2)*T:+.4f}，实测 {logS[:,-1].mean():+.4f} ✅")
    print(f"      虽然 μ=0，但因 -½σ²，价格中位数会【下漂】（中位数 {np.median(S_T):.3f} < 1）")
    print("   🔑 这就是第01课 simulate_gbm 里那行 (mu - 0.5*sigma**2) 的来历，")
    print("      也是第03课『+10% 再 -10% 不回本』的数学本质（来自伊藤引理/凸性）。\n")


def main():
    print("=" * 62)
    print("  随机过程 · 从抛硬币到布朗运动")
    print("=" * 62)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    exp1_random_walk(axes[0, 0], axes[0, 1])
    exp2_brownian_limit(axes[1, 0])
    exp3_ito_insight(axes[1, 1])
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "随机过程配图.png"
    fig.savefig(out, dpi=110)
    exp4_volatility_drag()
    print("=" * 62)
    print(f"  图已保存: {out.name}（随机游走/终点分布/布朗极限/伊藤秘密 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
