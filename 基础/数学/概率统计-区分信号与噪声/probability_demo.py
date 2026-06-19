"""
概率统计 · 用四个实验理解「信号 vs 噪声」
==========================================
概率统计是量化的灵魂。这份代码用四个动手实验，把抽象概念变成你能看到的图：

  实验①  大数定律：样本越多，平均值越接近真相（噪声被平均掉）
  实验②  中心极限定理：为什么「正态分布」无处不在
  实验③  假设检验：一个策略赚钱，是真本事还是运气？(t 检验 / p 值)
  实验④  多重检验陷阱：试得越多，越容易『撞』出假信号(呼应量化第05课)

运行：python probability_demo.py
依赖：numpy, scipy, matplotlib
注：图内文字用英文（避免缺中文字体时显示成方块）；讲解在终端输出与笔记.md 里。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

RNG = np.random.default_rng(0)
TD = 252  # 一年交易日


def exp1_lln(ax):
    """实验①：大数定律——掷骰子，看 running 平均如何收敛到真值 3.5。"""
    rolls = RNG.integers(1, 7, size=5000)
    running_mean = np.cumsum(rolls) / np.arange(1, len(rolls) + 1)
    ax.plot(running_mean, color="#1f77b4", lw=1)
    ax.axhline(3.5, color="red", ls="--", label="true mean 3.5")
    ax.set_title("(1) Law of Large Numbers: mean converges as n grows")
    ax.set_xlabel("number of dice rolls (log)")
    ax.set_ylabel("running mean")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("① 大数定律：掷骰子 5000 次")
    print(f"   前 10 次平均 = {running_mean[9]:.3f}，5000 次平均 = {running_mean[-1]:.3f}（真值 3.5）")
    print("   —— 单次充满随机(噪声)，但大量重复后平均值稳定(信号)。\n")


def exp2_clt(ax):
    """实验②：中心极限定理——把极不正态的均匀分布求平均，分布趋于正态。"""
    n_per_sample, n_samples = 30, 5000
    sample_means = RNG.uniform(0, 1, size=(n_samples, n_per_sample)).mean(axis=1)
    ax.hist(sample_means, bins=40, density=True, color="#2ca02c",
            edgecolor="white", alpha=0.8)
    mu, sigma = 0.5, np.sqrt(1 / 12 / n_per_sample)
    xs = np.linspace(sample_means.min(), sample_means.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), "r-", lw=2, label="theoretical normal")
    ax.set_title("(2) Central Limit Theorem: mean of uniforms -> normal")
    ax.set_xlabel("mean of 30 uniform draws")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("② 中心极限定理：对『完全不像钟形』的均匀分布取平均")
    print(f"   样本均值的分布：均值≈{sample_means.mean():.3f}，标准差≈{sample_means.std():.4f}")
    print("   —— 大量独立随机量的『和/平均』总是趋于正态。这就是正态无处不在的原因。\n")


def exp3_hypothesis_test(ax):
    """实验③：假设检验——一个『真有微弱优势』的策略，多少数据才能证明它？"""
    true_daily_edge = 0.0003   # 真实日均超额收益 0.03%（很微弱但为正）
    daily_vol = 0.01           # 日波动 1%
    years = 2
    n = years * TD
    returns = true_daily_edge + daily_vol * RNG.standard_normal(n)

    # 单样本 t 检验：H0 = 真实均值为 0（即没有优势，全是运气）
    t_stat, p_value = stats.ttest_1samp(returns, 0.0)
    sharpe = returns.mean() / returns.std() * np.sqrt(TD)

    equity = np.cumprod(1 + returns)
    ax.plot(equity, color="#9467bd")
    ax.set_title(f"(3) Real edge or luck?\nSharpe={sharpe:.2f}, t={t_stat:.2f}, p={p_value:.2f}")
    ax.set_xlabel("trading day")
    ax.set_ylabel("net value")
    ax.grid(alpha=0.3)

    print(f"③ 假设检验：一个真实日均优势 {true_daily_edge:.2%} 的策略，跑 {years} 年")
    print(f"   年化夏普 {sharpe:.2f}，t 统计量 {t_stat:.2f}，p 值 {p_value:.3f}")
    verdict = "能" if p_value < 0.05 else "【不能】"
    print(f"   p {'<' if p_value < 0.05 else '>'} 0.05 → {verdict}在统计上断定它真有优势。")
    print(f"   🔑 经验法则：t ≈ 夏普 × √年数。要让 t>2(显著)，")
    print(f"      夏普 0.5 的策略需约 {(2/0.5)**2:.0f} 年数据！这就是为什么『证明一个策略有效』极难。\n")


def exp4_multiple_testing(ax):
    """实验④：多重检验——测一堆『毫无优势』的随机策略，总有几个『碰巧显著』。"""
    n_strategies, n = 1000, 2 * TD
    rets = 0.01 * RNG.standard_normal((n_strategies, n))   # 每个都是纯噪声(真实均值=0)
    t_stats, p_values = stats.ttest_1samp(rets, 0.0, axis=1)
    n_false_pos = int((p_values < 0.05).sum())
    best = int(np.argmax(t_stats))                          # t 值最正的『赢家』
    best_sharpe = rets[best].mean() / rets[best].std() * np.sqrt(TD)

    ax.hist(p_values, bins=20, color="#ff7f0e", edgecolor="white")
    ax.axvline(0.05, color="red", ls="--", label="p = 0.05")
    ax.set_title(f"(4) {n_strategies} pure-noise strategies\n{n_false_pos} are 'significant' (p<0.05) by luck")
    ax.set_xlabel("p-value")
    ax.set_ylabel("number of strategies")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    print(f"④ 多重检验陷阱：测试 {n_strategies} 个【毫无优势】的纯噪声策略")
    print(f"   仍有 {n_false_pos} 个『碰巧』p<0.05 显著（≈5%，正是假阳性率）")
    print(f"   其中看起来『最牛』的那个，年化夏普 {best_sharpe:.2f}——但它纯粹是运气！")
    print("   —— 试得越多，越容易撞出假信号。这就是量化第05课『数据窥探』的统计本质。\n")


def main():
    print("=" * 60)
    print("  概率统计 · 用四个实验理解『信号 vs 噪声』")
    print("=" * 60)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    exp1_lln(axes[0, 0])
    exp2_clt(axes[0, 1])
    exp3_hypothesis_test(axes[1, 0])
    exp4_multiple_testing(axes[1, 1])
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "概率统计配图.png"
    fig.savefig(out, dpi=110)
    print("=" * 60)
    print(f"  图已保存: {out.name}（大数定律/中心极限/假设检验/多重检验 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
