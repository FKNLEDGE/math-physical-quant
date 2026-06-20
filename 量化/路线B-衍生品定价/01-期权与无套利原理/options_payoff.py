"""
B1 · 期权与无套利原理（四张收益图 + 平价验证）
==============================================
期权的价值，全部来自『到期时在每种情景下赔付多少』(收益图) +
『不能凭空赚无风险钱』(无套利)。这节课先把这两块直觉建起来。

  图①  看涨/看跌期权的『曲棍球杆』收益图
  图②  保护性看跌 = 给股票买保险（收益有了地板）
  图③  看跌看涨平价：两个组合到期收益完全相同 → 价格必相同(无套利)
  图④  跨式组合 = 同时买看涨+看跌 = 赌『大涨大跌』而非方向

运行：python options_payoff.py
图内文字用英文；讲解在终端与 笔记.md。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def call_payoff(S, K):
    return np.maximum(S - K, 0.0)   # 看涨：有权以 K 买入 → 赚 S-K（S>K 时）


def put_payoff(S, K):
    return np.maximum(K - S, 0.0)   # 看跌：有权以 K 卖出 → 赚 K-S（S<K 时）


def main():
    K = 100.0           # 行权价
    S0 = 100.0          # 当前股价
    r = 0.05            # 无风险年利率
    T = 1.0             # 到期 1 年
    S = np.linspace(50, 150, 400)   # 到期日各种可能的股价

    print("=" * 60)
    print("  B1 · 期权与无套利原理")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 看涨/看跌 收益
    ax = axes[0, 0]
    ax.plot(S, call_payoff(S, K), color="#d62728", lw=2, label="Call payoff: max(S-K,0)")
    ax.plot(S, put_payoff(S, K), color="#1f77b4", lw=2, label="Put payoff: max(K-S,0)")
    ax.axvline(K, color="gray", ls="--", lw=0.8)
    ax.set_title("(1) Option payoffs at expiry (hockey sticks)")
    ax.set_xlabel("stock price at expiry S_T"); ax.set_ylabel("payoff")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 保护性看跌 = 保险
    ax = axes[0, 1]
    ax.plot(S, S, color="#888", lw=1.5, ls=":", label="Stock alone")
    ax.plot(S, S + put_payoff(S, K), color="#2ca02c", lw=2, label="Stock + Put (insured)")
    ax.axhline(K, color="red", ls="--", lw=0.8, label=f"floor = K = {K:.0f}")
    ax.set_title("(2) Protective put = insurance (a floor on losses)")
    ax.set_xlabel("stock price at expiry S_T"); ax.set_ylabel("portfolio value")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图③ 看跌看涨平价：Call+Bond(K) 与 Put+Stock 收益相同
    ax = axes[1, 0]
    bond = np.full_like(S, K)                    # 到期拿到 K 现金的零息债
    portA = call_payoff(S, K) + bond             # 看涨 + 债券
    portB = put_payoff(S, K) + S                 # 看跌 + 股票
    ax.plot(S, portA, color="#d62728", lw=3, label="Call + Bond(K)")
    ax.plot(S, portB, color="#1f77b4", lw=1.5, ls="--", label="Put + Stock")
    ax.set_title("(3) Put-Call Parity: identical payoff -> identical price")
    ax.set_xlabel("stock price at expiry S_T"); ax.set_ylabel("payoff = max(S,K)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 跨式：看涨+看跌
    ax = axes[1, 1]
    straddle = call_payoff(S, K) + put_payoff(S, K)
    ax.plot(S, straddle, color="#9467bd", lw=2, label="Straddle = Call + Put")
    ax.axvline(K, color="gray", ls="--", lw=0.8)
    ax.set_title("(4) Straddle: bet on a BIG move (either direction)")
    ax.set_xlabel("stock price at expiry S_T"); ax.set_ylabel("payoff")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "B1配图.png"
    fig.savefig(out, dpi=110)

    # 文本：看跌看涨平价的『无套利』威力——不靠任何模型，一个价格锁死另一个
    print("① 期权收益（到期）：看涨 = max(S-K,0)，看跌 = max(K-S,0)")
    print("   只有『权利』没有『义务』，所以收益是『曲棍球杆』形（亏损封顶在权利金）。\n")
    print("② 保护性看跌：持股 + 买一份看跌 = 给股票买保险")
    print(f"   无论跌多狠，组合价值有地板 K={K:.0f}。代价是付一笔『保费』(看跌期权价)。\n")
    print("③ 看跌看涨平价（无套利的第一个漂亮结论）：")
    print("   组合A『看涨+面值K的债券』 与 组合B『看跌+股票』，到期收益都 = max(S,K)，完全相同。")
    print("   既然到期收益处处相同，今天的价格就必须相同（否则有人无风险套利）：")
    print("      C + K·e^(-rT) = P + S0")
    C_demo = 10.45
    P_implied = C_demo - S0 + K*np.exp(-r*T)
    print(f"   ▶ 举例：若看涨价 C={C_demo}，S0={S0:.0f}, K={K:.0f}, r={r:.0%}, T={T:.0f}：")
    print(f"     平价锁定看跌价 P = C - S0 + K·e^(-rT) = {P_implied:.2f}")
    print("     —— 注意：我们【没用任何定价模型】，仅凭『无套利』就把 P 钉死了！这就是无套利的威力。\n")
    print("④ 跨式：同时买看涨+看跌，收益是 V 形——大涨大跌都赚、横盘最亏。")
    print("   它赌的是『波动大小』，不是『方向』。呼应时间序列那课：方向难测、波动可做。\n")
    print(f"  图已保存: {out.name}（收益图/保险/平价/跨式 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
