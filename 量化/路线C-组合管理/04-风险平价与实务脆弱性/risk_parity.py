"""
C4 · 风险平价与实务脆弱性（路线 C 收尾）
========================================
预期收益 μ 最难估、噪声最大。风险平价干脆不估 μ，只按【风险】配置：
让每个资产对组合风险的贡献相等。更稳健，也更分散。

  实验①  风险贡献：60/40 的风险几乎全来自股票；风险平价把它拉平
  实验②  资本权重 ≠ 风险权重：60% 的钱在股票，却扛了 ~90% 的风险
  实验③  稳健性：扰动预期收益，最大夏普权重剧烈乱跳，风险类方法纹丝不动
  实验④  样本外赛马：1/N 和风险平价常常打平甚至胜过『最优化』的最大夏普

运行：python risk_parity.py
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

RNG = np.random.default_rng(3)
NAMES = ["Stocks", "Bonds", "Commod", "Credit"]
mu = np.array([0.08, 0.03, 0.07, 0.05])
sig = np.array([0.18, 0.06, 0.20, 0.10])
corr = np.array([
    [1.00, -0.20, 0.30, 0.50],
    [-0.20, 1.00, -0.10, 0.20],
    [0.30, -0.10, 1.00, 0.25],
    [0.50, 0.20, 0.25, 1.00],
])
Sigma = np.outer(sig, sig)*corr


def risk_contributions(w, S):
    """每个资产对组合波动的贡献（加总=组合波动）。"""
    port_vol = np.sqrt(w @ S @ w)
    mrc = S @ w / port_vol         # 边际风险贡献(导数!)
    return w * mrc                 # 风险贡献 = 权重 × 边际贡献


def w_min_var(S):
    inv = np.linalg.inv(S); ones = np.ones(len(S))
    w = inv @ ones / (ones @ inv @ ones)
    return np.clip(w, 0, None)/np.clip(w, 0, None).sum()


def w_max_sharpe(m, S, rf=0.02):
    inv = np.linalg.inv(S)
    w = inv @ (m - rf)
    return w/w.sum()


def w_risk_parity(S):
    """等风险贡献(ERC)：最小化各风险贡献的离散度。"""
    n = len(S)
    def obj(w):
        rc = risk_contributions(w, S)
        return np.sum((rc - rc.mean())**2)
    cons = [{"type": "eq", "fun": lambda w: w.sum()-1}]
    res = minimize(obj, np.ones(n)/n, bounds=[(1e-4, 1)]*n, constraints=cons, method="SLSQP")
    return res.x


def main():
    print("=" * 62)
    print("  C4 · 风险平价与实务脆弱性")
    print("=" * 62)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 四种方法的风险贡献
    ax = axes[0, 0]
    methods = {
        "60/40-ish": np.array([0.6, 0.4, 0.0, 0.0]),
        "min-var": w_min_var(Sigma),
        "risk-parity": w_risk_parity(Sigma),
        "equal 1/N": np.ones(4)/4,
    }
    x = np.arange(4); bw = 0.2
    for i, (name, w) in enumerate(methods.items()):
        rc = risk_contributions(w, Sigma); rc_pct = rc/rc.sum()
        ax.bar(x + (i-1.5)*bw, rc_pct, bw, label=name)
    ax.set_xticks(x); ax.set_xticklabels(NAMES)
    ax.set_title("(1) Risk contribution by asset (risk-parity = equal)")
    ax.set_ylabel("share of portfolio risk"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    rp = w_risk_parity(Sigma)
    print(f"① 风险平价权重 = {dict(zip(NAMES, np.round(rp,2)))}")
    print(f"   它的各资产风险贡献都≈25%（拉平）；而 60/40 的风险几乎全在股票。\n")

    # 图② 资本权重 vs 风险权重（60/40 股债）
    ax = axes[0, 1]
    S2 = Sigma[:2, :2]; w2 = np.array([0.6, 0.4])
    rc2 = risk_contributions(w2, S2); rc2 = rc2/rc2.sum()
    x2 = np.arange(2)
    ax.bar(x2-0.2, w2, 0.4, label="capital weight", color="#1f77b4")
    ax.bar(x2+0.2, rc2, 0.4, label="risk weight", color="#d62728")
    ax.set_xticks(x2); ax.set_xticklabels(["Stocks", "Bonds"])
    ax.set_title("(2) 60/40: 60% capital in stocks = ~90% of the RISK")
    ax.set_ylabel("share"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    print(f"② 经典 60/40：股票占资本 60%，却占风险 {rc2[0]:.0%}！'看似分散,实则全押股票'。\n")

    # 图③ 稳健性：扰动 μ，各方法权重的不稳定度
    ax = axes[1, 0]
    instab = {"max-Sharpe": [], "min-var": [], "risk-parity": [], "1/N": []}
    for _ in range(200):
        mu_n = mu + RNG.normal(0, 0.02, 4)
        instab["max-Sharpe"].append(w_max_sharpe(mu_n, Sigma))
        instab["min-var"].append(w_min_var(Sigma))
        instab["risk-parity"].append(w_risk_parity(Sigma))
        instab["1/N"].append(np.ones(4)/4)
    spreads = {k: np.mean(np.std(np.array(v), axis=0)) for k, v in instab.items()}
    ax.bar(spreads.keys(), spreads.values(), color=["#d62728", "#1f77b4", "#2ca02c", "#888"])
    ax.set_title("(3) Weight instability under 2% return-error")
    ax.set_ylabel("avg std of weights"); ax.grid(alpha=0.3, axis="y")
    print(f"③ 给 μ 加 2% 噪声，权重不稳定度：{ {k: round(float(v),3) for k,v in spreads.items()} }")
    print("   最大夏普剧烈乱跳；min-var/风险平价不吃 μ，纹丝不动(仅受Σ影响)。\n")

    # 图④ 样本外赛马
    ax = axes[1, 1]
    n_trials, n_in = 300, 60      # 每次用 60 个月样本估计
    oos = {"max-Sharpe": [], "min-var": [], "risk-parity": [], "1/N": []}
    L = np.linalg.cholesky(Sigma/12)        # 月度协方差
    for _ in range(n_trials):
        sample = (mu/12) + (L @ RNG.standard_normal((4, n_in))).T   # 60个月模拟收益
        mu_hat, S_hat = sample.mean(0)*12, np.cov(sample.T)*12
        ws = {"max-Sharpe": w_max_sharpe(mu_hat, S_hat), "min-var": w_min_var(S_hat),
              "risk-parity": w_risk_parity(S_hat), "1/N": np.ones(4)/4}
        for k, w in ws.items():
            oos[k].append((w @ mu)/np.sqrt(w @ Sigma @ w))          # 用真参数评估
    means = {k: np.mean(v) for k, v in oos.items()}
    ax.bar(means.keys(), means.values(), color=["#d62728", "#1f77b4", "#2ca02c", "#888"])
    ax.set_title("(4) Out-of-sample Sharpe: simple often wins")
    ax.set_ylabel("avg out-of-sample Sharpe"); ax.grid(alpha=0.3, axis="y")
    print(f"④ 样本外赛马(300次,60月估计) 平均真实夏普：{ {k: round(float(v),3) for k,v in means.items()} }")
    print("   『最优化』的最大夏普因吃了噪声 μ，样本外常垫底；1/N 和风险平价稳稳在前。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "C4配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（风险贡献/资本vs风险/稳健性/样本外赛马 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
