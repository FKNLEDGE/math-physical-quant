"""
时间序列 · 价格的记忆与平稳性（四个实验）
==========================================
时间序列 = 有先后顺序的数据，顺序本身含信息。核心问题：过去能预测未来吗？

  实验①  平稳性：价格(非平稳，均值漂移) vs 收益率(近平稳)
  实验②  收益率的『方向』几乎不可预测：自相关≈0（有效市场的指纹）
  实验③  但『大小』可预测：|收益率| 自相关>0（波动率聚集，金融最铁的规律）
  实验④  AR(1)：均值回归 / 纯噪声 / 动量，由一个参数 φ 决定

运行：python time_series.py
图内文字用英文；讲解在终端与 笔记.md。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(11)
TD = 252


def simulate_garch_returns(n, seed=11):
    """模拟带『波动率聚集 + 厚尾』的收益率（GARCH 风格）。"""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.10, 0.88
    sigma2 = np.full(n, 0.012**2)
    z = rng.standard_t(df=5, size=n) / np.sqrt(5/3)
    r = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + alpha*r[t-1]**2 + beta*sigma2[t-1]
        r[t] = np.sqrt(sigma2[t]) * z[t]
    return r


def acf(x, max_lag):
    """自相关函数：每个 lag 下，序列和它自己『错开 lag 步』的相关系数。"""
    x = np.asarray(x)
    return [np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(1, max_lag+1)]


def exp1_stationarity(ax):
    """实验①：价格(非平稳) vs 收益率(近平稳)，用滚动均值看。"""
    r = simulate_garch_returns(1500, seed=11)
    price = 100 * np.exp(np.cumsum(r))
    roll_mean = np.convolve(price, np.ones(60)/60, mode="valid")
    ax.plot(price, color="#1f77b4", lw=0.8, label="price (non-stationary)")
    ax.plot(np.arange(59, len(price)), roll_mean, color="red", lw=1.2, label="60d rolling mean")
    ax.set_title("(1) Price is non-stationary: its mean keeps drifting")
    ax.set_xlabel("day"); ax.set_ylabel("price")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("① 平稳性：价格 vs 收益率")
    print(f"   价格前半段均值 {price[:750].mean():.1f}，后半段 {price[750:].mean():.1f} —— 漂移大(非平稳)")
    print(f"   收益率前半段均值 {r[:750].mean():+.5f}，后半段 {r[750:].mean():+.5f} —— 都≈0(近平稳)")
    print("   🔑 只有平稳的东西才能建模/预测。所以第一步永远是：价格→收益率(差分)。\n")
    return r


def exp2_3_acf(ax, r):
    """实验②③：收益率自相关≈0，但 |收益率| 自相关>0（波动率聚集）。"""
    K = 25
    acf_r = acf(r, K)
    acf_absr = acf(np.abs(r), K)
    lags = np.arange(1, K+1)
    ci = 1.96 / np.sqrt(len(r))   # 95% 置信带，超出才算显著
    ax.bar(lags - 0.2, acf_r, width=0.4, color="#1f77b4", label="ACF of returns")
    ax.bar(lags + 0.2, acf_absr, width=0.4, color="#d62728", label="ACF of |returns|")
    ax.axhline(ci, color="gray", ls="--", lw=0.8); ax.axhline(-ci, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(2&3) Direction unpredictable, magnitude IS")
    ax.set_xlabel("lag (days)"); ax.set_ylabel("autocorrelation")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("②『方向』不可预测：收益率自相关")
    print(f"   lag1~5 自相关 ≈ {[round(float(v),3) for v in acf_r[:5]]} —— 都贴近 0(±{ci:.3f}置信带内)")
    print("   这就是『有效市场假说』的指纹：昨天涨不代表今天涨，没有免费的方向信号。")
    print("\n③『大小』可预测：|收益率| 自相关（波动率聚集）")
    print(f"   lag1~5 自相关 ≈ {[round(float(v),3) for v in acf_absr[:5]]} —— 显著为正！")
    print("   大波动后面常跟大波动(平静接平静)。这是金融最稳健的经验规律，GARCH 的根据。")
    print("   🔑 不对称：市场把『方向』藏得很好，却把『波动』泄露了 → 收益难测，但风险可测。\n")


def exp4_ar1(ax):
    """实验④：AR(1) x_t = φ x_{t-1} + ε，φ 决定均值回归/噪声/动量。"""
    n = 300
    eps = RNG.standard_normal(n)
    for phi, c, name in [(-0.7, "#1f77b4", "phi=-0.7 mean-revert"),
                         (0.0, "#2ca02c", "phi=0 pure noise"),
                         (0.9, "#d62728", "phi=0.9 momentum")]:
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi*x[t-1] + eps[t]
        ax.plot(x, color=c, lw=0.9, label=name)
    ax.set_title("(4) AR(1): one number phi sets the behavior")
    ax.set_xlabel("time"); ax.set_ylabel("x")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("④ AR(1)：x_t = φ·x_{t-1} + 噪声")
    print("   φ<0：均值回归(上去就想下来，对应『配对/价差套利』)")
    print("   φ=0：纯噪声(无记忆，随机游走的增量)")
    print("   φ>0：动量/趋势持续(对应『趋势跟踪』，第04课均线策略想抓的就是它)")
    print("   φ=1：就是随机游走本身(价格的模型)。φ 是一根『记忆旋钮』。\n")


def shuffle_test(r):
    """文本：打乱顺序，波动率聚集消失——证明那是真实的时间结构。"""
    lag1_orig = np.corrcoef(np.abs(r[:-1]), np.abs(r[1:]))[0, 1]
    rs = RNG.permutation(r)
    lag1_shuf = np.corrcoef(np.abs(rs[:-1]), np.abs(rs[1:]))[0, 1]
    print("【验证】把收益率随机打乱顺序，波动率聚集还在吗？")
    print(f"   原序列 |收益| 的 lag1 自相关 = {lag1_orig:+.3f}（有聚集）")
    print(f"   打乱后                      = {lag1_shuf:+.3f}（≈0，聚集消失）")
    print("   —— 聚集是『时间顺序』里的真实结构，不是数字本身的性质。顺序含信息。\n")


def main():
    print("=" * 62)
    print("  时间序列 · 价格的记忆与平稳性")
    print("=" * 62)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    r = exp1_stationarity(axes[0, 0])
    exp2_3_acf(axes[0, 1], r)
    exp4_ar1(axes[1, 1])
    # (1,0) 放收益率序列本身，肉眼看波动率聚集
    axes[1, 0].plot(r, color="#ff7f0e", lw=0.6)
    axes[1, 0].set_title("Returns: calm and stormy periods cluster")
    axes[1, 0].set_xlabel("day"); axes[1, 0].set_ylabel("daily return")
    axes[1, 0].grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "时间序列配图.png"
    fig.savefig(out, dpi=110)
    shuffle_test(r)
    print("=" * 62)
    print(f"  图已保存: {out.name}（平稳性/自相关/收益序列/AR(1) 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
