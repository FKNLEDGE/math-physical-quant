"""
进阶·物理味 ③ Hawkes 自激过程
================================
泊松过程假设事件【彼此独立、强度恒定】。但真实市场里,一个大跳会引发余震:
一笔大单触发更多单、一次违约触发连锁违约、一次暴跌引出更多暴跌——【事件会自我激发、扎堆】。

Hawkes(1971) 把这写进强度里:每发生一个事件,强度就【跳高 α】,然后【按 β 指数衰减】:

    λ(t) = μ + Σ_{t_i < t} α·e^{−β(t − t_i)}
            └基础率┘ └─── 历史事件的"余震"叠加 ───┘

  分支比 n = α/β = 平均每个事件能"生"几个孩子。n<1 稳定, n→1 临界(雪崩边缘), n≥1 爆炸。

  实验①  事件流:Hawkes 扎堆成簇 vs 泊松均匀撒点(同样的总事件数)
  实验②  强度 λ(t):每个事件让强度跳高 α、再指数衰减——自激的"余震"看得见
  实验③  分布:Hawkes 的每窗口事件数【过度离散】(忽多忽少),泊松方差≈均值
  实验④  分支比 n→1:聚集越来越剧烈(Fano 因子飙升),逼近【自组织临界】(闭环E3)

运行：python hawkes_self_exciting.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(5)


def simulate_hawkes(mu, alpha, beta, T, rng=RNG):
    """Ogata 稀疏化(thinning)模拟指数核 Hawkes 过程,返回事件时刻数组。"""
    events = []
    t = 0.0
    ev = np.array([])
    while True:
        M = mu + alpha * np.sum(np.exp(-beta * (t - ev)))   # 当前强度=接下来区间的上界(强度只会衰减)
        t += rng.exponential(1.0 / M)
        if t >= T:
            break
        lam = mu + alpha * np.sum(np.exp(-beta * (t - ev)))  # t 处真实强度(已衰减)
        if rng.random() <= lam / M:                          # 稀疏化:按 λ/M 接受
            events.append(t)
            ev = np.array(events)
    return np.array(events)


def intensity_path(events, mu, alpha, beta, grid):
    """在 grid 上计算 Hawkes 强度 λ(t)。"""
    lam = np.full(len(grid), mu)
    for i, t in enumerate(grid):
        past = events[events < t]
        if past.size:
            lam[i] += alpha * np.sum(np.exp(-beta * (t - past)))
    return lam


def fano_factor(events, T, nbins=50):
    """Fano 因子 = 每窗口事件数的 方差/均值。泊松=1;越大越扎堆(过度离散)。"""
    counts, _ = np.histogram(events, bins=np.linspace(0, T, nbins + 1))
    return counts.var() / counts.mean() if counts.mean() > 0 else np.nan


def main():
    print("=" * 60)
    print("  进阶·物理味 ③ Hawkes 自激过程")
    print("=" * 60)
    mu, alpha, beta, T = 0.6, 0.8, 1.0, 150.0
    n_branch = alpha / beta
    events = simulate_hawkes(mu, alpha, beta, T)
    base_rate = mu / (1 - n_branch)                          # Hawkes 的长期平均强度
    poisson = np.sort(RNG.uniform(0, T, len(events)))        # 同样事件数的泊松(均匀)
    print(f"  参数: μ={mu}, α={alpha}, β={beta} → 分支比 n=α/β={n_branch:.2f}")
    print(f"  长期平均强度 μ/(1−n)={base_rate:.2f}; 模拟出 {len(events)} 个事件\n")

    print(f"① Hawkes 共 {len(events)} 个事件,肉眼可见扎堆成簇;泊松同样多但均匀。")
    fano_h = fano_factor(events, T)
    fano_p = fano_factor(poisson, T)
    print(f"③ Fano 因子(方差/均值): Hawkes {fano_h:.1f} ≫ 泊松 {fano_p:.1f}(≈1)。Hawkes 过度离散=扎堆。\n")

    # 分支比扫描(图④)
    ns = np.linspace(0.1, 0.92, 12)
    fanos = []
    for nb in ns:
        fs = [fano_factor(simulate_hawkes(mu, nb * beta, beta, T), T) for _ in range(4)]
        fanos.append(np.nanmean(fs))
    fanos = np.array(fanos)
    print(f"④ 分支比 n 从 0.1→0.92, Fano 因子从 {fanos[0]:.1f} 飙到 {fanos[-1]:.1f}——")
    print("   越接近临界(n→1),聚集越剧烈(雪崩边缘),呼应 E3 自组织临界。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 事件流对比
    ax = axes[0, 0]
    ax.vlines(events, 0.55, 0.95, color="#d62728", lw=0.7)
    ax.vlines(poisson, 0.05, 0.45, color="#1f77b4", lw=0.7)
    ax.text(0.5, 1.0, "Hawkes (self-exciting): clustered", color="#d62728", fontsize=9)
    ax.text(0.5, 0.48, "Poisson (independent): uniform", color="#1f77b4", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_yticks([])
    ax.set_title("(1) Same #events: Hawkes clusters, Poisson spreads out")
    ax.set_xlabel("time"); ax.grid(alpha=0.3, axis="x")

    # 图② 强度路径
    ax = axes[0, 1]
    grid = np.linspace(0, T, 1500)
    lam = intensity_path(events, mu, alpha, beta, grid)
    ax.plot(grid, lam, color="#d62728", lw=0.9, label="λ(t) intensity")
    ax.axhline(mu, color="gray", ls="--", lw=1, label=f"base μ={mu}")
    ax.plot(events, np.full(len(events), mu * 0.5), "|", color="k", ms=6, label="events")
    ax.set_title("(2) Each event bumps intensity by α, then it decays (aftershocks)")
    ax.set_xlabel("time"); ax.set_ylabel("intensity λ(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图③ 每窗口事件数分布
    ax = axes[1, 0]
    nb = 50
    ch, _ = np.histogram(events, bins=np.linspace(0, T, nb + 1))
    cp, _ = np.histogram(poisson, bins=np.linspace(0, T, nb + 1))
    mx = max(ch.max(), cp.max())
    bins = np.arange(0, mx + 2) - 0.5
    ax.hist(ch, bins=bins, alpha=0.6, color="#d62728", label=f"Hawkes (Fano {fano_h:.1f})")
    ax.hist(cp, bins=bins, alpha=0.6, color="#1f77b4", label=f"Poisson (Fano {fano_p:.1f})")
    ax.set_title("(3) Events per window: Hawkes is over-dispersed (bursty)")
    ax.set_xlabel("events per time window"); ax.set_ylabel("count of windows")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 图④ 分支比 → Fano
    ax = axes[1, 1]
    ax.plot(ns, fanos, "o-", color="#9467bd", label="Hawkes Fano vs branching ratio")
    ax.axhline(1, color="#1f77b4", ls="--", lw=1.2, label="Poisson (Fano=1)")
    ax.axvline(1.0, color="#d62728", ls=":", lw=1.2, label="critical n→1 (avalanche)")
    ax.set_title("(4) Approaching criticality: clustering explodes as n→1")
    ax.set_xlabel("branching ratio n = α/β"); ax.set_ylabel("Fano factor (var/mean)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "Hawkes配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（事件流/强度/分布/临界 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
