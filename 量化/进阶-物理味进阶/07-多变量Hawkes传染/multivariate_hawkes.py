"""
进阶·物理味 ⑦ 多变量 Hawkes 传染网络（系统性风险）
====================================================
[③ Hawkes](../03-Hawkes自激过程/) 是【单资产】自激:一个事件触发同资产更多事件。
真实系统性风险是【跨资产/跨机构】的:一家银行违约触发别家、一只股票暴跌带崩一篮子。
多变量 Hawkes 把"自激"推广成【互激网络】:资产 j 的事件,会按矩阵 G[i,j] 激发资产 i 的事件。

  关键量:**分支矩阵 G 的谱半径 ρ(G)**(③ 标量分支比的矩阵版):
    ρ<1 稳定;ρ→1 临界——一个外部冲击会级联成【系统级雪崩】,放大倍数 = 1/(1−ρ)。

  实验①  4 个资产的事件流:一处冲击跨资产级联(传染肉眼可见)
  实验②  传染网络 G:谁激发谁(矩阵热图)
  实验③  系统放大 1/(1−ρ):耦合越强(ρ→1),一个冲击引发的总事件数爆炸(系统脆弱)
  实验④  临界附近级联规模【重尾】:多数小、偶有吞没全系统的巨灾(闭环 E3 自组织临界)

运行：python multivariate_hawkes.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def mv_hawkes(mu, G, beta, T, rng, track_cluster=False):
    """多变量 Hawkes(分支/簇表示)。G[i,j]=资产j一个事件在资产i的后代期望数。
    返回事件列表 [(时刻, 资产)];track_cluster=True 时另返回每个移民簇的规模。"""
    d = len(mu)
    events = []
    sizes = {}
    # 移民 + 队列(携带 root 簇 id)
    queue = []
    rid = 0
    for i in range(d):
        for tt in rng.uniform(0, T, rng.poisson(mu[i] * T)):
            events.append((tt, i)); queue.append((tt, i, rid)); sizes[rid] = 1; rid += 1
    while queue:
        t, j, root = queue.pop()
        for i in range(d):
            for _ in range(rng.poisson(G[i, j])):
                ct = t + rng.exponential(1 / beta)
                if ct < T:
                    events.append((ct, i)); queue.append((ct, i, root)); sizes[root] += 1
    if track_cluster:
        return events, np.array(list(sizes.values()))
    return events


def buildG(a, b, d=4):
    """对角自激 a、非对角互激 b 的分支矩阵。"""
    G = np.full((d, d), b); np.fill_diagonal(G, a)
    return G


def spectral_radius(G):
    return max(abs(np.linalg.eigvals(G)))


def main():
    print("=" * 60)
    print("  进阶·物理味 ⑦ 多变量 Hawkes 传染网络（系统性风险）")
    print("=" * 60)
    d, beta, T = 4, 1.0, 3000.0
    mu = np.full(d, 0.3)
    G = buildG(0.45, 0.13, d)               # 中等耦合,ρ≈0.84
    rho = spectral_radius(G)
    print(f"  {d} 个资产;分支矩阵谱半径 ρ(G) = {rho:.2f}（<1 稳定;→1 临界雪崩）\n")
    print(f"① 一处冲击会跨资产级联(传染);② 互激矩阵 G 决定谁激发谁\n")

    # ③ 系统放大 vs 谱半径
    rhos, amp_sim, amp_th = [], [], []
    for a, b in [(0.3, 0.03), (0.4, 0.06), (0.4, 0.1), (0.45, 0.12), (0.5, 0.13), (0.5, 0.15), (0.52, 0.15)]:
        Gk = buildG(a, b, d); rk = spectral_radius(Gk)
        ev, sizes = mv_hawkes(mu, Gk, beta, T, np.random.default_rng(0), track_cluster=True)
        rhos.append(rk); amp_sim.append(sizes.mean()); amp_th.append(1 / (1 - rk))
    print("③ 系统放大(一个冲击引发的平均总事件数):")
    for r, s, th in zip(rhos, amp_sim, amp_th):
        print(f"   ρ={r:.2f}: 模拟 {s:5.1f} ≈ 理论 1/(1−ρ)={th:5.1f}")
    print("   耦合越强(ρ→1),放大越猛——'太互联而不能倒'的脆弱性。\n")

    # ④ 临界附近级联规模分布
    Gc = buildG(0.5, 0.153, d); rc = spectral_radius(Gc)
    _, sizes_c = mv_hawkes(mu, Gc, beta, T, np.random.default_rng(1), track_cluster=True)
    print(f"④ 临界附近(ρ={rc:.2f})级联规模: 中位 {int(np.median(sizes_c))}, 最大 {sizes_c.max()}（重尾:多数小、偶有巨灾）\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # 图① 多资产事件流(取一窗看跨资产级联)
    ax = axes[0, 0]
    ev = mv_hawkes(mu, G, beta, T, np.random.default_rng(3))
    ev = [(t, i) for t, i in ev if t < 250]
    for t, i in ev:
        ax.vlines(t, i - 0.4, i + 0.4, color=colors[i], lw=0.7)
    ax.set_yticks(range(d)); ax.set_yticklabels([f"asset {i}" for i in range(d)])
    ax.set_title("(1) Cross-asset cascade: a burst in one spreads to others")
    ax.set_xlabel("time"); ax.grid(alpha=0.3, axis="x")

    # 图② 传染网络矩阵
    ax = axes[0, 1]
    im = ax.imshow(G, cmap="Reds", vmin=0)
    ax.set_xticks(range(d)); ax.set_xticklabels([f"a{j}" for j in range(d)])
    ax.set_yticks(range(d)); ax.set_yticklabels([f"a{i}" for i in range(d)])
    ax.set_xlabel("source (event in...)"); ax.set_ylabel("excites (events in...)")
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center",
                    color="white" if G[i, j] > 0.3 else "black", fontsize=9)
    ax.set_title("(2) Contagion network G (who excites whom)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 图③ 系统放大
    ax = axes[1, 0]
    order = np.argsort(rhos)
    rr = np.array(rhos)[order]
    ax.plot(rr, np.array(amp_sim)[order], "o", color="#d62728", ms=7, label="simulated")
    xs = np.linspace(0.3, 0.96, 100)
    ax.plot(xs, 1 / (1 - xs), "-", color="#1f77b4", lw=1.5, label="theory 1/(1−ρ)")
    ax.axvline(1.0, color="k", ls=":", lw=1, label="ρ=1 critical")
    ax.set_title("(3) Systemic amplification explodes as coupling ρ→1")
    ax.set_xlabel("spectral radius ρ(G)"); ax.set_ylabel("avg events per shock")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 级联规模重尾(CCDF, log-log)
    ax = axes[1, 1]
    s = np.sort(sizes_c)
    ccdf = 1 - np.arange(len(s)) / len(s)
    ax.loglog(s, ccdf, ".", ms=3, color="#9467bd")
    ax.set_title(f"(4) Near-critical (ρ={rc:.2f}) cascade sizes are heavy-tailed")
    ax.set_xlabel("cascade size (events triggered)"); ax.set_ylabel("P(size ≥ x)")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "多变量Hawkes配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（跨资产级联/传染网络/系统放大/重尾 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
