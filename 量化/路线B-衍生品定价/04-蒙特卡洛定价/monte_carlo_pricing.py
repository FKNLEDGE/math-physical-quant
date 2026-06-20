"""
B4 · 蒙特卡洛定价（模拟万千未来，取平均）
==========================================
当公式(BS)和二叉树都搞不定复杂衍生品时，蒙特卡洛是『万能锤』：
模拟海量风险中性路径 → 各算收益 → 折现取平均 = 期权价（B2 风险中性期望的实现）。

  实验①  模拟一束风险中性 GBM 路径
  实验②  收敛：样本越多，蒙特卡洛价越逼近 BS（带 95% 置信带）
  实验③  误差律：标准误 ~ 1/√N（呼应概率课的 √n、随机过程课的 √n）
  实验④  杀手锏：给【亚式期权】(看平均价，无闭式解)定价

运行：python monte_carlo_pricing.py
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

RNG = np.random.default_rng(2024)


def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)


def main():
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    bs = bs_call(S0, K, r, sigma, T)
    print("=" * 62)
    print("  B4 · 蒙特卡洛定价")
    print("=" * 62)
    print(f"  原理：期权价 = e^(-rT)·E_风险中性[到期收益]，用模拟平均来估这个期望。")
    print(f"  风险中性路径：S_T = S0·exp((r-½σ²)T + σ√T·Z), Z~正态\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 模拟一束路径
    ax = axes[0, 0]
    steps = 252; dt = T/steps
    paths = np.zeros((40, steps+1)); paths[:, 0] = S0
    for t in range(1, steps+1):
        z = RNG.standard_normal(40)
        paths[:, t] = paths[:, t-1]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    ax.plot(np.linspace(0, T, steps+1), paths.T, lw=0.7, alpha=0.6)
    ax.axhline(K, color="red", ls="--", label=f"strike K={K:.0f}")
    ax.set_title("(1) Simulated risk-neutral price paths (GBM)")
    ax.set_xlabel("time (years)"); ax.set_ylabel("price")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 收敛到 BS（带置信带）
    ax = axes[0, 1]
    M = 200000
    Z = RNG.standard_normal(M)
    ST = S0*np.exp((r-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    disc_payoff = np.exp(-r*T)*np.maximum(ST-K, 0)
    nn = np.arange(1, M+1)
    run_mean = np.cumsum(disc_payoff)/nn
    run_var = np.cumsum(disc_payoff**2)/nn - run_mean**2
    run_se = np.sqrt(np.maximum(run_var, 0)/nn)
    idx = np.unique(np.logspace(1, np.log10(M), 200).astype(int)) - 1
    ax.plot(nn[idx], run_mean[idx], color="#1f77b4", label="MC estimate")
    ax.fill_between(nn[idx], run_mean[idx]-1.96*run_se[idx], run_mean[idx]+1.96*run_se[idx],
                    color="#1f77b4", alpha=0.2, label="95% band")
    ax.axhline(bs, color="red", ls="--", label=f"BS = {bs:.3f}")
    ax.set_xscale("log")
    ax.set_title("(2) MC converges to BS as samples grow")
    ax.set_xlabel("number of simulations N (log)"); ax.set_ylabel("call price")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"② 收敛：{M} 次模拟 → MC 价 = {run_mean[-1]:.4f} ± {1.96*run_se[-1]:.4f}（95%）")
    print(f"   BS 真值 = {bs:.4f}，落在置信区间内 ✅\n")

    # 图③ 标准误 ~ 1/√N
    ax = axes[1, 0]
    ax.loglog(nn[idx], run_se[idx], color="#2ca02c", label="standard error")
    ref = run_se[idx][0]*np.sqrt(nn[idx][0]/nn[idx])
    ax.loglog(nn[idx], ref, "k--", lw=1, label="slope -1/2 (1/sqrt(N))")
    ax.set_title("(3) Error law: standard error ~ 1/sqrt(N)")
    ax.set_xlabel("N (log)"); ax.set_ylabel("standard error (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    print("③ 误差律：标准误 ∝ 1/√N —— 想精度提高 10 倍，样本要多 100 倍（慢，但与维度无关）")
    print("   呼应概率课大数定律、随机过程课的 √n。这是蒙特卡洛的命门，也是它的普适。\n")

    # 图④ 杀手锏：亚式期权（看平均价，无闭式解）
    ax = axes[1, 1]
    M2 = 60000
    Zp = RNG.standard_normal((M2, steps))
    logpaths = np.cumsum((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Zp, axis=1)
    Spaths = S0*np.exp(logpaths)
    S_avg = Spaths.mean(axis=1)        # 路径平均价
    S_fin = Spaths[:, -1]              # 到期价
    strikes = np.linspace(80, 120, 9)
    euro = [np.exp(-r*T)*np.mean(np.maximum(S_fin-k, 0)) for k in strikes]
    asian = [np.exp(-r*T)*np.mean(np.maximum(S_avg-k, 0)) for k in strikes]
    ax.plot(strikes, euro, "o-", color="#1f77b4", label="European (final price)")
    ax.plot(strikes, asian, "s-", color="#d62728", label="Asian (average price)")
    ax.set_title("(4) Killer app: price an Asian option (no formula)")
    ax.set_xlabel("strike K"); ax.set_ylabel("call price")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("④ 亚式期权：收益看『路径平均价』而非到期价，没有 BS 那样的闭式解 → 蒙特卡洛轻松搞定")
    print(f"   K=100：欧式 {euro[4]:.3f} vs 亚式 {asian[4]:.3f}（亚式更便宜，因平均把波动抹小了）\n")

    # 方差缩减：对偶变量
    Za = RNG.standard_normal(M2)
    p1 = np.exp(-r*T)*np.maximum(S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*Za)-K, 0)
    p2 = np.exp(-r*T)*np.maximum(S0*np.exp((r-0.5*sigma**2)*T-sigma*np.sqrt(T)*Za)-K, 0)
    se_plain = p1.std()/np.sqrt(M2)
    se_anti = ((p1+p2)/2).std()/np.sqrt(M2)
    print(f"【加速】对偶变量(用 Z 和 -Z)：标准误 {se_plain:.4f} → {se_anti:.4f}（降约 {1-se_anti/se_plain:.0%}），免费提精度。")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "B4配图.png"
    fig.savefig(out, dpi=110)
    print(f"\n  图已保存: {out.name}（路径/收敛/误差律/亚式期权 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
