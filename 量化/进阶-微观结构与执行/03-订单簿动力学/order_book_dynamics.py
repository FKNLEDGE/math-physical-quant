"""
进阶·微观结构与执行 ③ 订单簿动力学（从订单流到价格）
========================================================
[① 最优执行](../01-最优执行AlmgrenChriss/) 和 [② 做市](../02-做市与逆向选择/) 把订单簿当背景。
这一课让订单簿【自己动起来】:用三种最基本的订单流,模拟一个限价订单簿(LOB),
看【价格、买卖价差、排队】如何从微观订单流【涌现】出来。

三种事件(零智能/ZI 模型,Smith-Farmer 2003 一脉):
  · 限价单到达(挂在最优价附近,提供流动性)
  · 市价单到达(吃掉对手方最优档,消耗流动性)
  · 撤单(挂着的单被撤走)

  实验①  平均订单簿形状:深度随"离中间价的距离"——驼峰形(最优档反而薄)
  实验②  中间价路径:没有任何"基本面",价格纯粹从订单流的随机增删中涌现
  实验③  最优买价排队量随时间:被市价单吃空、又被限价单补上(排队动力学)
  实验④  价格变动的自相关:lag-1 为负=买卖价差弹跳(bid-ask bounce,经典微观结构事实)

运行：python order_book_dynamics.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_lob(r_lim=0.5, r_mkt=0.4, r_can=0.01, P=400, mid0=200,
                 iters=140000, warm=20000, seed=0):
    """事件驱动模拟限价订单簿。返回采样的 中间价/价差/最优买量 序列 + 平均深度剖面。"""
    rng = np.random.default_rng(seed)
    bid = np.zeros(P); ask = np.zeros(P)
    for p in range(mid0 - 10, mid0): bid[p] = 3
    for p in range(mid0 + 1, mid0 + 11): ask[p] = 3
    best_bid = lambda: (np.nonzero(bid)[0].max() if bid.any() else mid0 - 20)
    best_ask = lambda: (np.nonzero(ask)[0].min() if ask.any() else mid0 + 20)

    mids, spreads, bbq = [], [], []
    prof = np.zeros(31); nacc = 0
    for it in range(iters):
        bb, ba = best_bid(), best_ask()
        rest = bid.sum() + ask.sum()
        rates = np.array([r_lim, r_lim, r_mkt, r_mkt, min(r_can * rest, 0.5)])
        e = rng.choice(5, p=rates / rates.sum())
        g = rng.geometric(0.55)                       # 离最优价的格子数,多为 1~3
        if e == 0:                                    # 限价买:挂在卖一内侧(收窄价差),但不穿价
            bid[max(min(ba - g, ba - 1), 1)] += 1
        elif e == 1:                                  # 限价卖
            ask[min(max(bb + g, bb + 1), P - 2)] += 1
        elif e == 2:                                  # 市价买:吃掉卖一
            if ask[ba] > 0: ask[ba] -= 1
        elif e == 3:                                  # 市价卖:吃掉买一
            if bid[bb] > 0: bid[bb] -= 1
        else:                                         # 撤单:随机撤一个挂单
            av = np.concatenate([bid, ask])
            if av.sum() > 0:
                i = rng.choice(2 * P, p=av / av.sum())
                if i < P: bid[i] = max(bid[i] - 1, 0)
                else: ask[i - P] = max(ask[i - P] - 1, 0)
        if bid.sum() < 2: bid[best_ask() - 2] += 3    # 兜底:一侧过薄就补一档
        if ask.sum() < 2: ask[best_bid() + 2] += 3
        if it > warm and it % 4 == 0:
            bb, ba = best_bid(), best_ask(); m = (bb + ba) / 2
            mids.append(m); spreads.append(ba - bb); bbq.append(bid[bb])
            for d in range(-15, 16):
                pp = int(round(m)) + d
                if 0 <= pp < P: prof[d + 15] += bid[pp] + ask[pp]
            nacc += 1
    return (np.array(mids), np.array(spreads), np.array(bbq), prof / max(nacc, 1))


def main():
    print("=" * 60)
    print("  进阶·微观结构与执行 ③ 订单簿动力学")
    print("=" * 60)
    mids, spreads, bbq, prof = simulate_lob()
    inc = np.diff(mids)
    print(f"  模拟完成: 采样 {len(mids)} 点; 平均价差 {spreads.mean():.2f} 格(中位 {np.median(spreads):.0f})")
    print(f"① 订单簿形状: 驼峰形——最优档薄(常被吃空),深度峰值在离中间价几格处")
    print(f"② 中间价在 {mids.min():.0f}~{mids.max():.0f} 间游走——无基本面,价格纯由订单流涌现")
    print(f"③ 最优买价排队量 均值 {bbq.mean():.1f}: 被市价单吃空、被限价单补上")
    # ④ 价格变动自相关
    def acf(x, k):
        x = x - x.mean()
        return [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, k + 1)]
    ac = acf(inc, 10)
    print(f"④ 价格变动 lag-1 自相关 = {ac[0]:+.2f}(为负=买卖价差弹跳 bid-ask bounce,经典事实)\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    d = np.arange(-15, 16)
    ax.bar(d, prof, color=["#2ca02c" if x < 0 else "#d62728" if x > 0 else "#888" for x in d], alpha=0.8)
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_title("(1) Average book shape: hump (best level is thin)")
    ax.set_xlabel("ticks from mid (bid ← | → ask)"); ax.set_ylabel("avg depth"); ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.plot(mids, color="#1f77b4", lw=0.6)
    ax.set_title("(2) Mid-price emerges from order flow (no fundamentals)")
    ax.set_xlabel("sampled event"); ax.set_ylabel("mid price (ticks)"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    seg = slice(0, 1500)
    ax.plot(np.arange(seg.stop)[seg], bbq[seg], color="#ff7f0e", lw=0.8)
    ax.set_title("(3) Best-bid queue: depleted by market orders, refilled by limits")
    ax.set_xlabel("sampled event"); ax.set_ylabel("best-bid queue size"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar(np.arange(1, 11), ac, color="#9467bd")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"(4) Price-change autocorr: lag-1 = {ac[0]:.2f} (bid-ask bounce)")
    ax.set_xlabel("lag"); ax.set_ylabel("autocorrelation of Δmid"); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "订单簿配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（簿形状/中间价/排队/弹跳 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
