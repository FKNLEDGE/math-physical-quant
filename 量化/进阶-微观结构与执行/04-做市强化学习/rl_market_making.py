"""
进阶·微观结构与执行 ④ 做市强化学习（RL 重新发现 Avellaneda-Stoikov）
======================================================================
[② 做市](../02-做市与逆向选择/) 里,Avellaneda-Stoikov 用数学推出"按库存偏移报价"。
这一课换个思路:不给公式,让一个【强化学习(Q-learning)智能体】从零开始、靠试错,
自己【学会】该怎么报价——看它会不会【自发地重新发现】同一个"逆库存偏斜"策略。

  环境:中间价随机游走;智能体每步选报价倾向(偏买/中性/偏卖),按报价激进度成交;
       奖励 = 赚到的价差 − γ·库存²(持有库存要受罚=风险厌恶)。
  状态 = 当前库存(离散);动作 = {偏买, 中性, 偏卖}。

  实验①  学习曲线:训练中每轮奖励上升(智能体在变聪明)
  实验②  学到的策略:库存→动作——自发呈"逆库存偏斜"(库存多→偏卖,正是 AS!)
  实验③  库存路径:RL 智能体把库存摁在 0 附近 vs 傻瓜(永远中性)库存乱走
  实验④  盈亏分布:RL 的 PnL 比傻瓜【窄得多】(风险大降),呼应 ② 的均值-方差

运行：python rl_market_making.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

QMAX = 8
NS, NA = 2 * QMAX + 1, 3                 # 状态=库存(-8..8); 动作=0偏买/1中性/2偏卖
PB = [0.5, 0.35, 0.1]                    # 各动作的"买成交概率"(偏买时高)
PS = [0.1, 0.35, 0.5]                    # 各动作的"卖成交概率"(偏卖时高)
EDGE = 1.0                               # 每笔成交赚到的半价差


def train(gamma=0.08, episodes=8000, steps=200, alpha=0.2, eps0=0.25, grl=0.95, seed=0):
    """表格 Q-learning。奖励=成交价差 − γ·库存²。返回 Q 表与每轮总奖励。"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((NS, NA))
    hist = []
    for ep in range(episodes):
        q = 0; tot = 0.0
        eps = max(0.02, eps0 * (1 - ep / episodes))
        for _ in range(steps):
            s = q + QMAX
            a = rng.integers(NA) if rng.random() < eps else int(np.argmax(Q[s]))
            nf = 0
            if rng.random() < PB[a] and q < QMAX: q += 1; nf += 1
            if rng.random() < PS[a] and q > -QMAX: q -= 1; nf += 1
            r = nf * EDGE - gamma * q * q
            Q[s, a] += alpha * (r + grl * Q[q + QMAX].max() - Q[s, a])
            tot += r
        hist.append(tot)
    return Q, np.array(hist)


def rollout(policy, gamma=0.08, steps=200, n_ep=400, sigma=0.3, seed=1):
    """用给定策略跑评估:真实中间价随机游走,记录库存路径与终端 MtM 盈亏。
    policy: 函数 q->action。返回 (一条示例库存路径, 各轮盈亏数组)。"""
    rng = np.random.default_rng(seed)
    pnls = []; sample_inv = None
    for ep in range(n_ep):
        q = 0; cash = 0.0; S = 100.0; inv = [0]
        for _ in range(steps):
            a = policy(q)
            if rng.random() < PB[a] and q < QMAX:    # 买入成交:付 S-EDGE
                q += 1; cash -= S - EDGE
            if rng.random() < PS[a] and q > -QMAX:   # 卖出成交:收 S+EDGE
                q -= 1; cash += S + EDGE
            S += sigma * rng.standard_normal()       # 中间价随机游走(库存的风险来源)
            inv.append(q)
        pnls.append(cash + q * S)                    # 终端按市价结清库存
        if ep == 0: sample_inv = np.array(inv)
    return sample_inv, np.array(pnls)


def main():
    print("=" * 60)
    print("  进阶·微观结构与执行 ④ 做市强化学习（RL 重新发现 AS）")
    print("=" * 60)
    Q, hist = train()
    pol = np.argmax(Q, axis=1)
    name = {0: "偏买", 1: "中性", 2: "偏卖"}
    print("② 学到的策略(库存→动作):")
    for q in [-6, -3, -1, 0, 1, 3, 6]:
        print(f"   库存 q={q:+d} → {name[pol[q + QMAX]]}")
    print("   清晰的'逆库存偏斜':库存为正→偏卖、为负→偏买——RL 自发重现了 Avellaneda-Stoikov!\n")

    rl_policy = lambda q: int(pol[q + QMAX])
    naive_policy = lambda q: 1                       # 傻瓜:永远中性,不管库存
    inv_rl, pnl_rl = rollout(rl_policy)
    inv_nv, pnl_nv = rollout(naive_policy)
    print(f"③ 库存绝对峰值: RL {np.abs(inv_rl).max()} vs 傻瓜 {np.abs(inv_nv).max()}（RL 摁在 0 附近）")
    print(f"④ 盈亏: RL 均值 {pnl_rl.mean():+.1f} 标准差 {pnl_rl.std():.1f} | "
          f"傻瓜 均值 {pnl_nv.mean():+.1f} 标准差 {pnl_nv.std():.1f}")
    print("   RL 用'逆库存偏斜'把盈亏标准差(风险)大幅压低——和 ② 的均值-方差权衡一致。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    sm = np.convolve(hist, np.ones(100) / 100, mode="valid")
    ax.plot(sm, color="#1f77b4", lw=1)
    ax.set_title("(1) Learning curve: reward rises as the agent learns")
    ax.set_xlabel("episode"); ax.set_ylabel("episode reward (smoothed)"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    qs = np.arange(-QMAX, QMAX + 1)
    colors = ["#2ca02c" if a == 0 else "#888" if a == 1 else "#d62728" for a in pol]
    ax.bar(qs, pol, color=colors)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["lean BUY", "neutral", "lean SELL"])
    ax.set_title("(2) Learned policy = skew against inventory (rediscovers A-S!)")
    ax.set_xlabel("inventory q"); ax.grid(alpha=0.3, axis="y")
    ax.annotate("long → lean sell", xy=(5, 2), xytext=(1.5, 1.55), fontsize=8.5, color="#d62728")
    ax.annotate("short → lean buy", xy=(-5, 0), xytext=(-7.5, 0.45), fontsize=8.5, color="#2ca02c")

    ax = axes[1, 0]
    ax.plot(inv_nv, color="#d62728", lw=0.8, alpha=0.8, label=f"naive (always neutral), |q|max {np.abs(inv_nv).max()}")
    ax.plot(inv_rl, color="#1f77b4", lw=0.9, label=f"RL agent, |q|max {np.abs(inv_rl).max()}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(3) Inventory path: RL keeps it near 0, naive drifts")
    ax.set_xlabel("step"); ax.set_ylabel("inventory q"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bins = np.linspace(min(pnl_nv.min(), pnl_rl.min()), max(pnl_nv.max(), pnl_rl.max()), 30)
    ax.hist(pnl_nv, bins=bins, alpha=0.55, color="#d62728", label=f"naive (std {pnl_nv.std():.0f})")
    ax.hist(pnl_rl, bins=bins, alpha=0.6, color="#1f77b4", label=f"RL (std {pnl_rl.std():.0f})")
    ax.axvline(0, color="k", lw=1)
    ax.set_title("(4) P&L distribution: RL is much tighter (lower inventory risk)")
    ax.set_xlabel("terminal P&L"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "做市RL配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（学习曲线/策略/库存/盈亏 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
