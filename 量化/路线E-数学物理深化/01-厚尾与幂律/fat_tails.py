"""
E1 · 厚尾与幂律（正态的崩塌）
============================
主流金融建立在『高斯/布朗运动』上(随机过程课、B5)。但真实市场不是高斯的：
极端涨跌远比正态预测的频繁。这是 econophysics 的头号事实。

  实验①  真实收益分布 vs 正态：尖峰 + 厚尾(对数纵轴看得清)
  实验②  QQ 图：尾部偏离正态的直线 → 厚尾的指纹
  实验③  幂律尾：P(|r|>x) ~ x^(-α)，双对数下是直线(α≈3 反立方律)
  实验④  『不可能』的 sigma 事件：4σ、5σ 暴跌的实际次数 >> 正态预测

运行：python fat_tails.py
依赖：numpy, scipy, matplotlib（数据用 量化/数据/ 示例行情）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import quant_tools as qt

RNG = np.random.default_rng(0)


def main():
    print("=" * 60)
    print("  E1 · 厚尾与幂律（正态的崩塌）")
    print("=" * 60)
    # 真实(示例)收益
    r = qt.to_log_returns(qt.load_sample_data()["Close"]).values
    z = (r - r.mean())/r.std()                 # 标准化
    kurt = stats.kurtosis(z)                    # 超额峰度(正态=0)
    print(f"  示例收益超额峰度 = {kurt:.1f}（正态=0；>0 即尖峰厚尾）")
    print(f"  真实股市超额峰度常达 5~50，1987黑色星期一是约 -20σ 事件(正态下几乎不可能)。\n")

    # 大样本厚尾模型(Student-t, df=3 → 幂律指数 α=3, 即'反立方律')
    big = stats.t(df=3).rvs(200000, random_state=1)
    big = big/big.std()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 分布对比(对数纵轴)
    ax = axes[0, 0]
    xs = np.linspace(-6, 6, 300)
    ax.hist(z, bins=80, density=True, alpha=0.6, color="#1f77b4", label="real returns (sample)")
    ax.plot(xs, stats.norm.pdf(xs), "r-", lw=2, label="Gaussian (same std)")
    ax.set_yscale("log")
    ax.set_title("(1) Real returns: peaked center + FAT TAILS")
    ax.set_xlabel("standardized return (sigma)"); ax.set_ylabel("density (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("① 对数纵轴下：中间比正态更尖，两端('尾部')比正态高出几个数量级——厚尾。\n")

    # 图② QQ 图
    ax = axes[0, 1]
    stats.probplot(z, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(3)
    ax.set_title("(2) QQ-plot: tails bend away from normal line")
    ax.grid(alpha=0.3)
    print("② QQ图：若是正态,点应落在直线上；实际两端'翘起/下垂'偏离 → 厚尾的标准指纹。\n")

    # 图③ 幂律尾(双对数)
    ax = axes[1, 0]
    for data, name, c in [(big, "fat-tailed (t, df=3)", "#d62728"),
                          (RNG.standard_normal(200000), "Gaussian", "#1f77b4")]:
        a = np.sort(np.abs(data))[::-1]
        surv = np.arange(1, len(a)+1)/len(a)        # P(|X|>x)
        ax.loglog(a[::50], surv[::50], ".", ms=3, color=c, label=name)
    # 拟合幂律指数(用厚尾数据的尾部)
    tail = np.sort(np.abs(big))[::-1][:4000]
    xv = np.log(tail); yv = np.log(np.arange(1, len(tail)+1)/len(big))
    alpha = -np.polyfit(xv, yv, 1)[0]
    ax.set_title(f"(3) Power-law tail: straight on log-log (alpha~{alpha:.1f})")
    ax.set_xlabel("|return| (log)"); ax.set_ylabel("P(|X|>x) (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    print(f"③ 幂律尾：厚尾数据在双对数下尾部是【直线】(斜率≈-{alpha:.1f}=幂律指数α)；")
    print("   正态则迅速【向下弯】(尾部指数级衰减)。真实股市 α≈3,即'反立方律'(Stanley等)。\n")

    # 图④ sigma 事件：实际 vs 正态预测
    ax = axes[1, 1]
    ks = [3, 4, 5]
    n = len(big)
    obs = [(np.abs(big) > k).sum() for k in ks]
    exp = [n*2*(1-stats.norm.cdf(k)) for k in ks]
    x = np.arange(len(ks)); bw = 0.35
    ax.bar(x-bw/2, obs, bw, color="#d62728", label="observed (fat-tailed)")
    ax.bar(x+bw/2, exp, bw, color="#1f77b4", label="Gaussian prediction")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{k}σ" for k in ks])
    ax.set_title("(4) 'Impossible' sigma events happen way more often")
    ax.set_ylabel("count in 200k days (log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    print("④ sigma 事件(20万天)：")
    for k, o, e in zip(ks, obs, exp):
        print(f"   {k}σ: 厚尾实际 {o} 次 vs 正态预测 {e:.1f} 次（差 {o/max(e,1e-9):.0f} 倍）")
    print("   正态把尾部风险系统性地严重低估 → VaR/Black-Scholes 在危机时失灵(LTCM,2008)。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "E1配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（分布/QQ/幂律尾/sigma事件 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
