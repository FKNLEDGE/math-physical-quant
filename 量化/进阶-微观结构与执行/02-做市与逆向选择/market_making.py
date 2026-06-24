"""
进阶·微观结构与执行 ② 做市与逆向选择
========================================
① 讲的是【吃流动性】的人(你要把大单买进卖出);这一课讲【提供流动性】的人——
做市商(market maker)。他挂买卖价、赚价差,但要面对两个敌人:

  逆向选择(Glosten-Milgrom 1985): 和他成交的人里,有些是【知情者】(知道真实价值)。
      做市商分不清谁是知情、谁是噪声,只能把价差拉宽来补偿被知情者"挑走"的亏损。
      → 价差的本质,是【信息不对称】的代价。订单流还让做市商【贝叶斯更新】对真值的估计。

  库存风险(Avellaneda-Stoikov 2008): 做市商攒了一堆多头/空头(库存),价格一动就亏。
      最优做法:按库存【偏移报价】(保留价 r 随库存反向移动)→ 逼着库存均值回归到 0。

  实验①  Glosten-Milgrom: 订单流让做市商的真值估计收敛,买卖价始终把真值夹在中间
  实验②  价差 vs 知情者比例: 知情者越多→价差越宽(逆向选择的代价)
  实验③  Avellaneda-Stoikov: 保留价随库存反向偏移→报价偏斜→把库存往 0 推
  实验④  库存控制对决: AS 偏斜报价(库存稳在0附近) vs 对称报价(库存乱走=高风险)

运行：python market_making.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(11)


# ==================== 一、Glosten-Milgrom ====================

def gm_quotes(p, alpha, VH, VL):
    """给定真值为高的后验 p、知情者比例 alpha,算做市商的 卖价(ask) 和 买价(bid)。
    ask=E[V|来了买单], bid=E[V|来了卖单]——必须如此才能不被知情者套利。"""
    pb_H = alpha * 1 + (1 - alpha) * 0.5     # P(买单|V高)
    pb_L = (1 - alpha) * 0.5                  # P(买单|V低)
    ps_H = (1 - alpha) * 0.5                  # P(卖单|V高)
    ps_L = alpha * 1 + (1 - alpha) * 0.5      # P(卖单|V低)
    Pbuy = p * pb_H + (1 - p) * pb_L
    Psell = p * ps_H + (1 - p) * ps_L
    ask = (p * pb_H * VH + (1 - p) * pb_L * VL) / Pbuy
    bid = (p * ps_H * VH + (1 - p) * ps_L * VL) / Psell
    return ask, bid, pb_H, pb_L


def gm_simulate(alpha=0.3, VH=101.0, VL=99.0, n=70, true_high=True, rng=RNG):
    """模拟一串订单流(真值=VH),记录做市商后验 p、买卖价的收敛。"""
    p = 0.5
    asks, bids, mids = [], [], []
    for _ in range(n):
        ask, bid, pb_H, pb_L = gm_quotes(p, alpha, VH, VL)
        asks.append(ask); bids.append(bid); mids.append(p * VH + (1 - p) * VL)
        # 生成一笔订单:知情者(看真值买/卖) 或 噪声(50/50)
        if rng.random() < alpha:
            buy = true_high                  # 知情者:真值高→买
        else:
            buy = rng.random() < 0.5
        # 贝叶斯更新后验 p
        if buy:
            p = p * pb_H / (p * pb_H + (1 - p) * pb_L)
        else:
            p = p * (0.5 * (1 - alpha)) / (p * (0.5 * (1 - alpha)) + (1 - p) * (alpha + 0.5 * (1 - alpha)))
    return np.array(asks), np.array(bids), np.array(mids)


# ==================== 二、Avellaneda-Stoikov ====================

def as_simulate(skew=True, gamma=0.6, sigma=2.0, k=1.5, A=60.0, T=1.0, N=400, rng=None):
    """做市商模拟。skew=True 用 Avellaneda-Stoikov 按库存偏移报价;False=对称报价。
    返回 (时间, 库存路径, 终端PnL)。"""
    rng = rng or np.random.default_rng(0)
    dt = T / N
    S = 100.0
    q = 0          # 库存
    cash = 0.0
    qs = np.zeros(N + 1)
    for i in range(N):
        tau = T - i * dt
        if skew:
            r = S - q * gamma * sigma**2 * tau                 # 保留价随库存反向偏移
            half = 0.5 * gamma * sigma**2 * tau + (1 / gamma) * np.log(1 + gamma / k)
            d_bid = S - (r - half)                             # 我方买价离中价的距离
            d_ask = (r + half) - S
        else:
            d_bid = d_ask = 0.5 * (1 / k)                      # 对称、固定的半价差
        lam_b = A * np.exp(-k * d_bid) * dt                    # 买单打到我方买价→我buy
        lam_a = A * np.exp(-k * d_ask) * dt                    # 卖单打到我方卖价→我sell
        if rng.random() < lam_b:
            q += 1; cash -= (S - d_bid)
        if rng.random() < lam_a:
            q -= 1; cash += (S + d_ask)
        S += sigma * np.sqrt(dt) * rng.standard_normal()
        qs[i + 1] = q
    pnl = cash + q * S                                          # 终端按市价结清库存
    return np.linspace(0, T, N + 1), qs, pnl


def main():
    print("=" * 60)
    print("  进阶·微观结构与执行 ② 做市与逆向选择")
    print("=" * 60)
    VH, VL = 101.0, 99.0

    # ① GM 收敛
    asks, bids, mids = gm_simulate(alpha=0.3, VH=VH, VL=VL, n=70, true_high=True)
    print(f"① Glosten-Milgrom: 真值={VH}(高)。做市商初始估值 {mids[0]:.2f},价差 {asks[0]-bids[0]:.3f}")
    print(f"   70 笔订单后估值收敛到 {mids[-1]:.2f}→真值,买卖价始终把真值夹在中间。\n")

    # ② 价差 vs 知情者比例
    alphas = np.linspace(0.0, 0.95, 40)
    spreads = [gm_quotes(0.5, a, VH, VL)[0] - gm_quotes(0.5, a, VH, VL)[1] for a in alphas]
    print(f"② 价差 vs 知情者比例 α: α=0→价差 {spreads[0]:.2f}(无逆向选择);"
          f"α=0.95→价差 {spreads[-1]:.2f}。知情者越多,价差越宽。\n")

    # ③ AS 报价偏斜 vs 库存
    qs_grid = np.arange(-10, 11)
    gamma, sigma, k, tau = 0.6, 2.0, 1.5, 0.5
    half = 0.5 * gamma * sigma**2 * tau + (1 / gamma) * np.log(1 + gamma / k)
    res_offset = -qs_grid * gamma * sigma**2 * tau            # 保留价相对中价的偏移
    print(f"③ Avellaneda-Stoikov: 库存为正(多头)→保留价下移→报价偏低→更易卖出→库存回 0。\n")

    # ④ 库存控制对决(多次模拟)
    inv_as, inv_sym, pnl_as, pnl_sym = [], [], [], []
    for s in range(300):
        rng = np.random.default_rng(s)
        _, q_as, p_as = as_simulate(skew=True, rng=rng)
        rng = np.random.default_rng(s)
        _, q_sy, p_sy = as_simulate(skew=False, rng=rng)
        inv_as.append(np.abs(q_as).max()); inv_sym.append(np.abs(q_sy).max())
        pnl_as.append(p_as); pnl_sym.append(p_sy)
    print(f"④ 库存控制(300次模拟, 价格波动较大时库存风险才显形):")
    print(f"   AS 偏斜报价 : 库存绝对峰值 {np.mean(inv_as):.1f} | PnL 均值 {np.mean(pnl_as):+.1f} 标准差 {np.std(pnl_as):.1f}")
    print(f"   对称报价    : 库存绝对峰值 {np.mean(inv_sym):.1f} | PnL 均值 {np.mean(pnl_sym):+.1f} 标准差 {np.std(pnl_sym):.1f}")
    print("   AS 把库存摁在 0 附近→PnL 标准差小得多(风险大降);代价是少赚一点均值——")
    print("   这正是 AS 要最优化的【均值-方差权衡】:不追最高收益,追最好的风险调整后收益。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    xx = np.arange(len(asks))
    ax.plot(xx, asks, color="#d62728", lw=1.2, label="ask")
    ax.plot(xx, bids, color="#2ca02c", lw=1.2, label="bid")
    ax.plot(xx, mids, color="#1f77b4", lw=1.5, label="MM belief E[V]")
    ax.axhline(VH, color="k", ls="--", lw=1, label=f"true value {VH:.0f}")
    ax.fill_between(xx, bids, asks, color="gray", alpha=0.15)
    ax.set_title("(1) Glosten-Milgrom: order flow → belief converges to true value")
    ax.set_xlabel("trade #"); ax.set_ylabel("price"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(alphas, spreads, color="#9467bd", lw=2)
    ax.set_title("(2) Spread comes from adverse selection")
    ax.set_xlabel("fraction of informed traders α"); ax.set_ylabel("bid-ask spread")
    ax.annotate("more informed →\nwider spread", xy=(0.8, spreads[32]), xytext=(0.35, spreads[-1]*0.6),
                fontsize=9, color="#9467bd", arrowprops=dict(arrowstyle="->", color="#9467bd"))
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(qs_grid, res_offset + half, "o-", color="#d62728", ms=3, label="ask offset from mid")
    ax.plot(qs_grid, res_offset - half, "o-", color="#2ca02c", ms=3, label="bid offset from mid")
    ax.plot(qs_grid, res_offset, "--", color="#1f77b4", label="reservation price offset")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_title("(3) Avellaneda-Stoikov: quotes skew against inventory")
    ax.set_xlabel("inventory q (long →)"); ax.set_ylabel("quote offset from mid")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bins = np.linspace(min(min(pnl_as), min(pnl_sym)), max(max(pnl_as), max(pnl_sym)), 30)
    ax.hist(pnl_sym, bins=bins, alpha=0.55, color="#d62728", label=f"symmetric (PnL std {np.std(pnl_sym):.1f})")
    ax.hist(pnl_as, bins=bins, alpha=0.55, color="#1f77b4", label=f"AS skewed (PnL std {np.std(pnl_as):.1f})")
    ax.axvline(0, color="k", lw=1)
    ax.set_title("(4) AS controls inventory → much tighter P&L distribution")
    ax.set_xlabel("terminal P&L (300 runs)"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "做市配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（GM收敛/价差与逆向选择/AS偏斜/库存控制 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
