"""
B2 · 二叉树与风险中性（用复制+无套利真刀真枪定价）
==================================================
B1 用无套利钉住了『价格关系』。这节课用『复制』钉住单个期权的『绝对价格』，
并引出衍生品定价最深的概念——风险中性定价。

  实验①  单期复制：用 Δ 股股票 + 借钱，造出与期权一模一样的收益 → 期权价=复制成本
  实验②  震撼：真实涨跌概率 p 根本不进入定价！(风险中性概率 q 才是主角)
  实验③  多期二叉树：层层向后归纳，给期权定价
  实验④  取极限：步数 N→∞，二叉树价格收敛到 Black-Scholes

运行：python binomial_pricing.py
图内文字用英文；讲解在终端与 笔记.md。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def bs_call(S0, K, r, sigma, T):
    """Black-Scholes 看涨价（B3 会推导；这里先用作『收敛目标』参照）。"""
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)


def binomial_call(S0, K, r, sigma, T, N):
    """N 步 CRR 二叉树给欧式看涨定价（向后归纳）。"""
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt)); d = 1/u
    q = (np.exp(r*dt) - d) / (u - d)        # 风险中性概率
    j = np.arange(N+1)
    ST = S0 * u**j * d**(N-j)               # 到期各节点股价
    C = np.maximum(ST - K, 0.0)             # 到期收益
    disc = np.exp(-r*dt)
    for _ in range(N):                      # 一层层往回折现期望
        C = disc * (q*C[1:] + (1-q)*C[:-1])
    return C[0]


def one_period_replication():
    """实验①②：单期复制定价 + 真实概率无关。"""
    S0, K, r, T = 100.0, 100.0, 0.05, 1.0
    u, d = 1.2, 0.9
    Su, Sd = S0*u, S0*d
    Cu, Cd = max(Su-K, 0), max(Sd-K, 0)     # 看涨到期收益

    # 复制：解 Δ·Su + B·e^{rT}=Cu, Δ·Sd + B·e^{rT}=Cd
    Delta = (Cu - Cd) / (Su - Sd)
    B = np.exp(-r*T) * (Cu - Delta*Su)
    price_repl = Delta*S0 + B

    # 风险中性概率
    q = (np.exp(r*T) - d) / (u - d)
    price_rn = np.exp(-r*T) * (q*Cu + (1-q)*Cd)

    print("① 单期复制定价（S0=100, 上涨到120 或 下跌到90, K=100, r=5%）")
    print(f"   到期：涨→股价120 期权赚 {Cu:.0f}；跌→股价90 期权赚 {Cd:.0f}")
    print(f"   复制组合：买 Δ={Delta:.4f} 股股票 + 借 {-B:.2f} 元现金")
    print(f"   → 这个组合到期收益和期权【完全一样】，所以期权价 = 复制成本 = {price_repl:.4f}")
    print(f"   风险中性概率 q = (e^rT - d)/(u-d) = {q:.4f}")
    print(f"   折现风险中性期望 = e^(-rT)[q·{Cu:.0f}+(1-q)·{Cd:.0f}] = {price_rn:.4f}  （与复制法一致 ✅）\n")

    print("② 震撼：真实涨跌概率 p 根本不影响期权价！")
    print(f"   {'真实涨概率p':>10} | {'天真折现期望(用p)':>16} | {'无套利复制价':>12}")
    for p in [0.2, 0.5, 0.8]:
        naive = np.exp(-r*T)*(p*Cu + (1-p)*Cd)
        print(f"   {p:>10.1f} | {naive:>16.4f} | {price_repl:>12.4f}")
    print("   → 左列随 p 乱变(还是错的)；右列恒定。定价用的是 q(风险中性)，不是 p(真实)！")
    print("   直觉：复制组合对冲掉了风险，所以你对涨跌的『主观看法 p』根本不该影响价格。\n")
    return price_repl


def main():
    print("=" * 62)
    print("  B2 · 二叉树与风险中性")
    print("=" * 62)

    one_period_replication()

    # 多期与收敛
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    bs = bs_call(S0, K, r, sigma, T)
    print(f"③④ 多期二叉树 → 收敛到 Black-Scholes（S0=K=100, r=5%, σ=20%, T=1）")
    for N in [1, 2, 5, 10, 50, 200]:
        print(f"   N={N:>4} 步：二叉树价 = {binomial_call(S0,K,r,sigma,T,N):.4f}")
    print(f"   Black-Scholes 真值       = {bs:.4f}  ← 步数越多，二叉树越逼近它\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 单期树示意
    ax = axes[0, 0]
    ax.annotate("S0=100\n(C=?)", (0, 0), fontsize=10, ha="center",
                bbox=dict(boxstyle="round", fc="#eef"))
    ax.annotate("Su=120\nCu=20", (1, 1), fontsize=10, ha="center",
                bbox=dict(boxstyle="round", fc="#efe"))
    ax.annotate("Sd=90\nCd=0", (1, -1), fontsize=10, ha="center",
                bbox=dict(boxstyle="round", fc="#fee"))
    ax.plot([0, 1], [0, 1], "k-"); ax.plot([0, 1], [0, -1], "k-")
    ax.set_xlim(-0.4, 1.5); ax.set_ylim(-1.6, 1.6)
    ax.set_title("(1) One-period tree: replicate to price")
    ax.axis("off")

    # 图② 真实概率无关
    ax = axes[0, 1]
    S0b, Kb, rb, Tb, u, d = 100, 100, 0.05, 1.0, 1.2, 0.9
    Cu, Cd = 20, 0
    Delta = (Cu-Cd)/(S0b*u - S0b*d); B = np.exp(-rb*Tb)*(Cu-Delta*S0b*u)
    repl = Delta*S0b + B
    ps = np.linspace(0.05, 0.95, 50)
    naive = np.exp(-rb*Tb)*(ps*Cu + (1-ps)*Cd)
    ax.plot(ps, naive, color="#d62728", label="naive: discount E[payoff] under real p")
    ax.axhline(repl, color="#1f77b4", lw=2, label=f"no-arbitrage price = {repl:.2f} (flat!)")
    ax.set_title("(2) Real probability p does NOT enter the price")
    ax.set_xlabel("real-world up probability p"); ax.set_ylabel("option price")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图③ 多期树节点(股价格子)
    ax = axes[1, 0]
    N = 6; dt = T/N; u = np.exp(sigma*np.sqrt(dt)); d = 1/u
    for i in range(N+1):
        for j in range(i+1):
            ax.scatter(i, S0*u**j*d**(i-j), color="#9467bd", s=20)
    ax.set_title("(3) Multi-period binomial lattice of stock prices")
    ax.set_xlabel("time step"); ax.set_ylabel("stock price")
    ax.grid(alpha=0.3)

    # 图④ 收敛到 BS
    ax = axes[1, 1]
    Ns = np.arange(1, 101)
    prices = [binomial_call(S0, K, r, sigma, T, n) for n in Ns]
    ax.plot(Ns, prices, color="#2ca02c", lw=1, label="binomial price")
    ax.axhline(bs, color="red", ls="--", label=f"Black-Scholes = {bs:.3f}")
    ax.set_title("(4) Binomial -> Black-Scholes as steps grow")
    ax.set_xlabel("number of steps N"); ax.set_ylabel("call price")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "B2配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（单期树/概率无关/多期格子/收敛BS 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
