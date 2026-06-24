"""
进阶·微观结构与执行 ① 最优执行（从订单簿到 Almgren-Chriss）
================================================================
前面所有课都假设你能"按收盘价"成交。真实世界里,价格不是一个数,而是一摞【订单簿】;
你的单子一进去就【推动价格】(冲击);买一大笔,必须在【冲击成本】和【价格风险】之间权衡。
Almgren-Chriss(2000) 把"怎么把大单拆开、按什么节奏执行"变成一道干净的最优化题。

  实验①  订单簿与价差:价格不是一个数,市价大单要【吃穿好几档】,均价比中间价差(滑点)
  实验②  价格冲击:你的大单把价格推走——【临时冲击】(成交后回弹)+【永久冲击】(留下不走)
  实验③  执行的有效前沿:快执行(冲击大/风险小) vs 慢执行(冲击小/风险大),一条均值-方差前沿
  实验④  Almgren-Chriss 最优轨迹:风险中性→匀速(TWAP 直线);越厌恶风险→越前置(早卖快卖)

运行：python optimal_execution.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(7)


# ==================== 一、订单簿 ====================

def build_book(mid=100.0, tick=0.01, spread_ticks=2, levels=8):
    """造一个示意限价订单簿:中间价两侧各 levels 档,越远档位挂单量越大。"""
    half = spread_ticks / 2 * tick
    ask_px = mid + half + tick * np.arange(levels)
    bid_px = mid - half - tick * np.arange(levels)
    ask_sz = (RNG.integers(2, 9, levels) * 100).astype(float)
    bid_sz = (RNG.integers(2, 9, levels) * 100).astype(float)
    return bid_px, bid_sz, ask_px, ask_sz, mid


def walk_book(ask_px, ask_sz, Q):
    """一笔市价买单 Q 股,从最优卖价开始【逐档吃上去】。返回成交均价和各档吃掉的量。"""
    filled = 0.0
    cost = 0.0
    used = np.zeros_like(ask_sz)
    for i, (p, s) in enumerate(zip(ask_px, ask_sz)):
        take = min(s, Q - filled)
        used[i] = take
        cost += take * p
        filled += take
        if filled >= Q:
            break
    return cost / filled, used


# ==================== 二、Almgren-Chriss ====================

def ac_trajectory(X, T, N, kappa):
    """Almgren-Chriss 最优持仓轨迹 x(t):风险厌恶强度通过 kappa 体现。
    kappa=0  → 直线匀速清仓(TWAP,风险中性);
    kappa>0  → x(t)=X·sinh(κ(T−t))/sinh(κT),前置(早卖快卖)。"""
    tj = np.linspace(0, T, N + 1)
    if kappa < 1e-8:
        x = X * (1 - tj / T)
    else:
        x = X * np.sinh(kappa * (T - tj)) / np.sinh(kappa * T)
    return tj, x


def ac_cost_var(X, T, N, sigma, eta, lam):
    """给定风险厌恶 lam,算最优执行的【期望临时冲击成本】与【成本方差(风险)】。
    kappa=√(lam·σ²/η)。返回 (E_cost, Var, 轨迹)。"""
    kappa = np.sqrt(lam * sigma**2 / eta) if lam > 0 else 0.0
    tj, x = ac_trajectory(X, T, N, kappa)
    tau = T / N
    n = -np.diff(x)                       # 每个区间卖出的股数(正)
    v = n / tau                           # 交易速率
    E_cost = eta * np.sum(v**2) * tau     # 临时冲击成本 ∫ η v² dt
    Var = sigma**2 * np.sum(x[1:]**2) * tau   # 持仓风险 ∫ σ² x² dt
    return E_cost, Var, tj, x, kappa


def main():
    print("=" * 64)
    print("  进阶·微观结构与执行 ① 最优执行（订单簿 → Almgren-Chriss）")
    print("=" * 64)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    # ---- 图① 订单簿与价差 ----
    bid_px, bid_sz, ask_px, ask_sz, mid = build_book()
    Q = 2600.0                              # 一笔市价买单(故意大到要吃几档)
    vwap, used = walk_book(ask_px, ask_sz, Q)
    spread = ask_px[0] - bid_px[0]
    slip = vwap - mid
    print(f"① 订单簿: 中间价 {mid:.2f}, 买卖价差 {spread*100:.0f} 分; 最优卖价 {ask_px[0]:.2f}")
    print(f"   一笔 {Q:.0f} 股的市价买单吃穿 {int((used>0).sum())} 档, 成交均价 {vwap:.4f}")
    print(f"   滑点(均价−中间价) = {slip*100:.2f} 分/股——价格不是一个数,大单要付穿档代价。\n")

    ax = axes[0, 0]
    ax.bar(bid_px, bid_sz, width=0.008, color="#2ca02c", label="bids (buy orders)")
    ax.bar(ask_px, ask_sz, width=0.008, color="#d62728", alpha=0.35, label="asks (sell orders)")
    ax.bar(ask_px, used, width=0.008, color="#d62728", label="asks eaten by market buy")
    ax.axvline(mid, color="k", lw=1, ls=":", label=f"mid {mid:.2f}")
    ax.axvline(vwap, color="#1f77b4", lw=1.6, label=f"fill VWAP {vwap:.3f}")
    ax.annotate("", xy=(ask_px[0], 880), xytext=(bid_px[0], 880),
                arrowprops=dict(arrowstyle="<->", color="purple"))
    ax.text(mid, 900, f"spread {spread*100:.0f}c", ha="center", color="purple", fontsize=8)
    ax.set_title("(1) The order book: price is a stack, not one number")
    ax.set_xlabel("price"); ax.set_ylabel("size (shares)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3, axis="y")

    # ---- 图② 价格冲击:临时 vs 永久 ----
    t = np.arange(140)
    s0, peak, perm = 100.0, 1.0, 0.6
    es, ee = 25, 80
    impact = np.zeros(len(t))
    for i in t:
        if i < es:
            impact[i] = 0.0
        elif i <= ee:
            impact[i] = peak * np.sqrt((i - es) / (ee - es))     # 执行中:凹形累积(平方根律)
        else:
            impact[i] = perm + (peak - perm) * np.exp(-(i - ee) / 14)  # 执行后:临时部分回弹
    price = s0 + impact + 0.02 * np.cumsum(RNG.standard_normal(len(t)))
    ax = axes[0, 1]
    ax.axvspan(es, ee, color="#1f77b4", alpha=0.08, label="your buying window")
    ax.plot(t, price, color="#1f77b4", lw=1.3)
    ax.axhline(s0, color="gray", ls=":", lw=1, label="arrival price")
    ax.axhline(s0 + perm, color="#2ca02c", ls="--", lw=1.2, label="permanent impact (stays)")
    ax.annotate("temporary\n(reverts)", xy=(ee, s0 + peak), xytext=(ee + 18, s0 + peak - 0.05),
                fontsize=8, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_title("(2) Price impact: you push the price (temporary + permanent)")
    ax.set_xlabel("time"); ax.set_ylabel("price"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    print("② 价格冲击: 买入时价格被你推高(执行中凹形累积=平方根律);")
    print("   成交后【临时冲击】回弹,但留下【永久冲击】不走——这就是 Almgren-Chriss 要权衡的成本。\n")

    # ---- 图③④ Almgren-Chriss ----
    X, T, N, sigma, eta = 1.0, 1.0, 60, 1.0, 1.0
    lams = np.concatenate([[0.0], np.geomspace(0.2, 60, 40)])
    Es, Vs = [], []
    for lam in lams:
        E, V, *_ = ac_cost_var(X, T, N, sigma, eta, lam)
        Es.append(E); Vs.append(V)
    Es, Vs = np.array(Es), np.array(Vs)

    # 选三个代表性风险厌恶:中性 / 中等 / 激进
    show = [(0.0, "#2ca02c", "λ=0  TWAP (risk-neutral)"),
            (4.0, "#1f77b4", "λ=4  moderate"),
            (36.0, "#d62728", "λ=36  aggressive (risk-averse)")]

    ax = axes[1, 0]
    ax.plot(Vs, Es, "-", color="#888", lw=1.5, label="efficient frontier")
    for lam, c, lab in show:
        E, V, *_ , kappa = ac_cost_var(X, T, N, sigma, eta, lam)
        ax.scatter([V], [E], s=80, color=c, zorder=5, label=lab)
    ax.set_title("(3) Execution efficient frontier (mean-variance)")
    ax.set_xlabel("variance of cost  (timing risk) →"); ax.set_ylabel("expected impact cost →")
    ax.annotate("trade SLOWER:\nless impact, more risk", xy=(Vs[0]*0.95, Es[0]), xytext=(Vs[0]*0.45, Es[0]+0.25),
                fontsize=8, color="#2ca02c", arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    ax.annotate("trade FASTER:\nmore impact, less risk", xy=(Vs[-1], Es[-1]), xytext=(Vs[-1]*0.15, Es[-1]*0.92),
                fontsize=8, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    print("④ Almgren-Chriss 最优轨迹(剩余持仓 x(t)):")
    for lam, c, lab in show:
        E, V, tj, x, kappa = ac_cost_var(X, T, N, sigma, eta, lam)
        ax.plot(tj, x, "-", color=c, lw=2, label=lab + f"  (κT={kappa*T:.1f})")
        print(f"   {lab:34s} 期望成本 {E:.3f} | 风险(方差) {V:.3f} | κT={kappa*T:.2f}")
    ax.set_title("(4) Optimal liquidation trajectories x(t)")
    ax.set_xlabel("time t / T"); ax.set_ylabel("shares remaining x(t)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("   风险中性(λ=0)=匀速直线(TWAP);越厌恶风险=越前置(早卖快卖,用冲击成本买'少担惊受怕')。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "最优执行配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（订单簿/价格冲击/有效前沿/最优轨迹 四合一）")
    print("=" * 64)


if __name__ == "__main__":
    main()
