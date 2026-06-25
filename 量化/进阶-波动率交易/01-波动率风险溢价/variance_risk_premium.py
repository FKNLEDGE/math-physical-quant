"""
进阶·波动率交易 ① 波动率风险溢价与"卖保险"策略
==================================================
[路线 B](../../路线B-衍生品定价/) 定价了期权,却没【交易】过它。这一课讲期权交易里最核心、也最危险的一件事:

  波动率风险溢价(VRP):**隐含波动率(IV)系统性高于事后已实现波动率(RV)。**
  因为期权是"保险",大家愿意为它【多付钱】——卖期权/卖波动率的人,长期收取这个溢价。

  但天下没有免费午餐:卖波动率 = 卖保险 = 平时稳收小钱,危机时赔上巨款。
  这就是著名的"**在压路机前捡钢镚**(picking up pennies in front of a steamroller)":
  胜率高、夏普好看,但偏度极负、左尾能要命。

  实验①  IV vs RV:多数时候 IV>RV(收溢价),但危机时 RV 暴过 IV(你赔大钱)
  实验②  卖波动率净值:稳稳爬升,被几次崩盘狠狠回吐(钢镚 vs 压路机)
  实验③  单期盈亏分布:均值为正,但【左尾肥、负偏度】——一次崩盘吃掉几十次盈利
  实验④  "看着很美"的陷阱:高胜率/好夏普,掩盖了"最坏一期=几十倍平均盈利"的尾部

运行：python variance_risk_premium.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

WIN = 21
ANN = np.sqrt(252)


def garch_with_crashes(n=1200, seed=3):
    """GARCH 波动聚集 + 偶发危机冲击,造出'平时温和、偶尔暴雷'的收益。"""
    rng = np.random.default_rng(seed)
    vol = np.zeros(n); vol[0] = 0.01; r = np.zeros(n)
    crash = np.zeros(n, bool)
    for t in range(1, n):
        spike = 0.05 if rng.random() < 0.004 else 0.0     # 偶发危机
        if spike > 0:
            crash[t] = True
        vol[t] = np.sqrt(2e-6 + 0.90 * vol[t - 1]**2 + 0.08 * r[t - 1]**2) + spike
        r[t] = vol[t] * rng.standard_normal()
    return r, crash


def main():
    print("=" * 60)
    print("  进阶·波动率交易 ① 波动率风险溢价与卖方策略")
    print("=" * 60)
    r, crash = garch_with_crashes()
    n = len(r)
    m = n - WIN
    rv_future = np.array([r[t:t + WIN].std() * ANN for t in range(m)])
    rv_past = np.array([r[max(t - WIN, 2):t].std() * ANN if t > 4 else 0.1 for t in range(m)])
    IV = np.clip(rv_past, 0.05, None) * 1.12              # 隐含≈过去已实现×1.12(经验溢价~10-15%)

    warm = 25
    pnl = (IV**2 - rv_future**2)[warm:]                  # 卖方差互换月度盈亏 ∝ IV²−RV²
    eq = np.cumsum(pnl)
    vrp = (IV - rv_future)[warm:]
    win_rate = (vrp > 0).mean()
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252 / WIN)
    skew = ((pnl - pnl.mean())**3).mean() / pnl.std()**3
    print(f"① VRP: IV 比 RV 平均高 {vrp.mean()*100:+.1f} 个 vol 点;{win_rate:.0%} 的时间 IV>RV(你收溢价)")
    print(f"② 卖波动率净值: 稳爬到峰值 {eq.max():.1f},几次崩盘回吐到 {eq[-1]:.1f}")
    print(f"③ 盈亏分布: 偏度 {skew:+.2f}(负=左尾肥);最坏一期亏 {pnl.min():.2f} = 平均盈利的 {abs(pnl.min()/pnl.mean()):.0f} 倍")
    print(f"④ '看着很美'的陷阱: 胜率 {win_rate:.0%}、年化夏普 {sharpe:.2f}——但一次崩盘吃掉几十次盈利")
    print("   卖波动率=卖保险=在压路机前捡钢镚。夏普骗你,尾部要命。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    idx = np.arange(warm, m)

    ax = axes[0, 0]
    ax.plot(idx, IV[warm:] * 100, color="#1f77b4", lw=0.9, label="implied vol (sold)")
    ax.plot(idx, rv_future[warm:] * 100, color="#d62728", lw=0.9, label="realized vol (paid)")
    ax.fill_between(idx, IV[warm:] * 100, rv_future[warm:] * 100,
                    where=rv_future[warm:] > IV[warm:], color="#d62728", alpha=0.3, label="RV>IV: you LOSE")
    ax.set_title("(1) IV usually > RV (you collect premium), except in crises")
    ax.set_xlabel("time"); ax.set_ylabel("volatility %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(idx, eq, color="#2ca02c", lw=1.2)
    dd = eq - np.maximum.accumulate(eq)
    crashes = idx[np.where(np.diff(np.concatenate([[0], dd])) < -0.3)[0]]
    ax.scatter(crashes, eq[crashes - warm], s=30, color="#d62728", zorder=5, label="vol-spike losses")
    ax.set_title("(2) Short-vol equity: steady climb, brutal crash drawdowns")
    ax.set_xlabel("time"); ax.set_ylabel("cumulative P&L"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(pnl, bins=60, color="#9467bd", edgecolor="white")
    ax.axvline(pnl.mean(), color="#2ca02c", lw=2, label=f"mean {pnl.mean():.3f} (positive)")
    ax.axvline(pnl.min(), color="#d62728", lw=2, ls="--", label=f"worst {pnl.min():.2f} (steamroller)")
    ax.set_yscale("log")
    ax.set_title(f"(3) P&L distribution: positive mean, fat left tail (skew {skew:.2f})")
    ax.set_xlabel("per-period P&L"); ax.set_ylabel("count (log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bars = ax.bar(["avg gain\n(good period)", "worst loss\n(crash)"], [pnl.mean(), pnl.min()],
                  color=["#2ca02c", "#d62728"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"(4) The trap: win {win_rate:.0%}, Sharpe {sharpe:.2f} — but one crash = {abs(pnl.min()/pnl.mean()):.0f}× a gain")
    ax.set_ylabel("P&L"); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, [pnl.mean(), pnl.min()]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "波动率溢价配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（IV-RV/净值/盈亏分布/陷阱 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
