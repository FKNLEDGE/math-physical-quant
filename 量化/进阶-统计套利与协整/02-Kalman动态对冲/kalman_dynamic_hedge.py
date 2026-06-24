"""
进阶·统计套利 ② Kalman 滤波动态对冲
======================================
①的协整配对有个硬伤:对冲比率 β 用 OLS 估【一次】,然后假设它【永远不变】。
真实世界里两只股票的关系会【漂移】(行业轮动、基本面变化)→ 固定 β 的价差慢慢
变得不平稳,均值回归策略被趋势反复打脸。

Kalman 滤波:把 β(和截距 α)当成【会随机游走的隐藏状态】,每来一个新数据点就
【在线更新】对它的估计——对冲比率随时间自适应。这正是你物理/工程里的最优线性估计器。

  状态  θ_t=[α_t, β_t]  服从随机游走(过程噪声 Q)
  观测  y_t = α_t + β_t·x_t + 噪声(观测噪声 R)

  实验①  β 真值在漂移:Kalman 紧紧跟住;静态 β(只用早期训练段估一次)卡在旧值
  实验②  价差:静态价差被 β 失配拖成趋势(不平稳);Kalman 价差始终平稳绕 0
  实验③  Kalman 标准化价差(创新/√S)=干净的均值回归信号,带进出场带
  实验④  公平对决(都因果、都年化夏普):跑 80 个随机实现,Kalman 稳定盈利,静态≈赔钱

运行：python kalman_dynamic_hedge.py
依赖：numpy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TD = 252
TRAIN = 150          # 静态 β 的训练段长度(也作为两者的预热/起跑点)


def make_drifting_pair(rng, n=750):
    """造一对'协整但对冲比率在漂移'的价格:y = α + β(t)·x + 平稳(OU)价差。"""
    x = 50 + np.cumsum(rng.normal(0, 0.7, n))          # x 价格(随机游走)
    beta_true = np.linspace(1.0, 1.7, n)               # β 真值线性漂移 1.0→1.7
    alpha_true = 5.0
    spread = np.zeros(n)                               # 平稳价差=可交易的部分
    for t in range(1, n):
        spread[t] = 0.92 * spread[t - 1] + rng.normal(0, 0.5)
    y = alpha_true + beta_true * x + spread
    return x, y, beta_true


def kalman_hedge(x, y, delta=1e-4, R=0.5):
    """对 [α, β] 做 Kalman 滤波(因果/在线)。delta 控制 β 允许漂移多快,R=观测噪声。
    返回 滤波后的 α,β、创新 e、创新方差 S。"""
    n = len(x)
    Q = delta / (1 - delta) * np.eye(2)
    theta = np.array([0.0, 1.0])
    P = np.eye(2) * 1.0
    alphas, betas, es, Ss = (np.zeros(n) for _ in range(4))
    for t in range(n):
        H = np.array([1.0, x[t]])
        P_pred = P + Q
        e = y[t] - H @ theta                          # 创新(预测误差)=当前价差
        S = H @ P_pred @ H + R
        K = (P_pred @ H) / S                          # 卡尔曼增益
        theta = theta + K * e
        P = P_pred - np.outer(K, H) @ P_pred
        alphas[t], betas[t], es[t], Ss[t] = theta[0], theta[1], e, S
    return alphas, betas, es, Ss


def rolling_z(s, w=60):
    """因果滚动 z-score(只用过去 w 个点),杜绝用全样本均值/方差的前视偏差。"""
    z = np.zeros(len(s))
    for t in range(w, len(s)):
        win = s[t - w:t]
        sd = win.std()
        z[t] = (s[t] - win.mean()) / sd if sd > 0 else 0.0
    return z


def trade_pair(z, y, x, beta_series, entry=1.0, exit=0.2):
    """对标准化价差 z 做均值回归,P&L 用【真实价格单位】结算(公平):
    持有'多 1 份 y、空 β 份 x',次日 P&L = pos·(Δy − β·Δx)。返回 (持仓, 每步PnL)。"""
    n = len(z)
    pos = np.zeros(n)
    for t in range(1, n):
        p = pos[t - 1]
        if p == 0:
            if z[t] > entry: p = -1
            elif z[t] < -entry: p = 1
        else:
            if abs(z[t]) < exit: p = 0
        pos[t] = p
    dy = np.diff(y, append=y[-1])
    dx = np.diff(x, append=x[-1])
    return pos, pos * (dy - beta_series * dx)


def sharpe(step_pnl):
    sd = step_pnl.std()
    return step_pnl.mean() / sd * np.sqrt(TD) if sd > 0 else 0.0


def eval_one(seed):
    """一个随机实现:返回 (静态夏普, Kalman夏普)。两者都因果、都同样的交易规则。"""
    rng = np.random.default_rng(seed)
    x, y, _ = make_drifting_pair(rng)
    n = len(x)
    a_tr, b_tr = np.linalg.lstsq(np.vstack([np.ones(TRAIN), x[:TRAIN]]).T, y[:TRAIN], rcond=None)[0]
    zs = rolling_z(y - (a_tr + b_tr * x))
    _, betas, es, Ss = kalman_hedge(x, y)
    zk = es / np.sqrt(Ss)
    b = TRAIN
    _, pnl_s = trade_pair(zs[b:], y[b:], x[b:], np.full(n, b_tr)[b:])      # 静态:卡死的训练段 β
    _, pnl_k = trade_pair(zk[b:], y[b:], x[b:], betas[b:])                # Kalman:时变 β
    return sharpe(pnl_s), sharpe(pnl_k)


def main():
    print("=" * 60)
    print("  进阶·统计套利 ② Kalman 滤波动态对冲")
    print("=" * 60)

    # ---- 单个实现(seed=3)用于图①②③ ----
    rng = np.random.default_rng(3)
    x, y, beta_true = make_drifting_pair(rng)
    n = len(x)
    a_tr, b_tr = np.linalg.lstsq(np.vstack([np.ones(TRAIN), x[:TRAIN]]).T, y[:TRAIN], rcond=None)[0]
    spread_static = y - (a_tr + b_tr * x)
    z_static = rolling_z(spread_static)
    alphas, betas, es, Ss = kalman_hedge(x, y)
    spread_kalman = es
    z_kalman = es / np.sqrt(Ss)
    b = TRAIN

    print(f"① β 真值 {beta_true[0]:.2f}→{beta_true[-1]:.2f};静态只用前 {TRAIN} 天估一次 β={b_tr:.2f}(卡死),")
    print(f"   Kalman 在线跟到末端 β={betas[-1]:.2f}。对真值平均误差 静态 "
          f"{np.mean(np.abs(b_tr-beta_true[b:])):.2f} vs Kalman {np.mean(np.abs(betas[b:]-beta_true[b:])):.2f}\n")
    print(f"② 价差(交易段)标准差: 静态 {spread_static[b:].std():.1f}(被拖成趋势) "
          f"vs Kalman {spread_kalman[b:].std():.2f}(平稳绕0)\n")

    _, pnl_s = trade_pair(z_static[b:], y[b:], x[b:], np.full(n, b_tr)[b:])
    _, pnl_k = trade_pair(z_kalman[b:], y[b:], x[b:], betas[b:])

    # ---- 80 个随机实现用于图④ ----
    sh = np.array([eval_one(s) for s in range(80)])
    sh_s, sh_k = sh[:, 0], sh[:, 1]
    print(f"④ 80 个随机实现的年化夏普(都因果、都同规则):")
    print(f"   静态对冲 : 均值 {sh_s.mean():+.2f} | 盈利比例 {(sh_s>0).mean():.0%}（β漂移后≈赔钱/掷硬币）")
    print(f"   Kalman   : 均值 {sh_k.mean():+.2f} | 盈利比例 {(sh_k>0).mean():.0%}（稳定盈利）")
    print("   结论:β 会漂时,静态对冲不是'差一点',而是系统性失效;动态对冲是必需品。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    idx = np.arange(n)

    ax = axes[0, 0]
    ax.plot(idx, beta_true, color="#7f7f7f", lw=2.5, label="true β (drifting)")
    ax.plot(idx, betas, color="#1f77b4", lw=1.3, label="Kalman β (adapts online)")
    ax.axhline(b_tr, color="#d62728", ls="--", lw=1.5, label=f"static β={b_tr:.2f} (train-only, stale)")
    ax.axvline(TRAIN, color="k", lw=0.8, ls=":", alpha=0.6, label="train/trade split")
    ax.set_title("(1) Hedge ratio: Kalman tracks the drift, static stays stale")
    ax.set_xlabel("time"); ax.set_ylabel("β"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(idx[b:], spread_static[b:], color="#d62728", lw=0.9, label="static-hedge spread (drifts → untradeable)")
    ax.plot(idx[b:], spread_kalman[b:], color="#1f77b4", lw=0.9, label="Kalman spread (stationary ~0)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(2) Spread: static drifts as β diverges, Kalman stays ~0")
    ax.set_xlabel("time"); ax.set_ylabel("spread"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(idx[b:], z_kalman[b:], color="#1f77b4", lw=0.8)
    for lvl, c, lab in [(1, "#2ca02c", "entry ±1"), (-1, "#2ca02c", None),
                        (0.2, "#ff7f0e", "exit ±0.2"), (-0.2, "#ff7f0e", None)]:
        ax.axhline(lvl, color=c, ls="--", lw=1, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(3) Kalman standardized spread = clean mean-reversion signal")
    ax.set_xlabel("time"); ax.set_ylabel("z = innovation / √S"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bins = np.linspace(min(sh_s.min(), sh_k.min()) - 0.1, max(sh_s.max(), sh_k.max()) + 0.1, 24)
    ax.hist(sh_s, bins=bins, alpha=0.6, color="#d62728", label=f"static (mean {sh_s.mean():.2f})")
    ax.hist(sh_k, bins=bins, alpha=0.6, color="#1f77b4", label=f"Kalman (mean {sh_k.mean():.2f})")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(sh_s.mean(), color="#d62728", ls="--", lw=1.5)
    ax.axvline(sh_k.mean(), color="#1f77b4", ls="--", lw=1.5)
    ax.set_title("(4) Fair fight over 80 runs: dynamic hedge reliably wins")
    ax.set_xlabel("annualized Sharpe"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "Kalman配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（β跟踪/价差/信号/夏普分布 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
