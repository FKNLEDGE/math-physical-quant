"""
第 01 课 · 你的第一段量化代码
================================
这段代码做了量化里最基础的三件事：
  1. 造一段「股价」—— 用几何布朗运动（GBM）模拟。
     彩蛋：布朗运动正是 1900 年 Bachelier 给期权定价、1905 年爱因斯坦解释
     花粉乱动用的同一套数学（见 docs/03-哲学与历史背景.md）。
  2. 算收益率 —— 从「价格」变成「每天涨跌百分之几」，这是量化的原材料。
  3. 算并画出关键指标 —— 年化收益、年化波动率、夏普比率。

运行：  python hello_quant.py
依赖：  numpy, pandas, matplotlib （见项目根目录 requirements.txt）

小白练习（跑通后做）：
  - 改下面的 MU（趋势）和 SIGMA（波动率），重跑，看价格曲线怎么变。
  - 把 SEED 改成别的数字，相当于「换一个平行世界」，再看一遍。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示器也能出图（存成文件）；本地有界面可删掉这行
import matplotlib.pyplot as plt


# ---------- 可调参数（练习就改这里） ----------
S0 = 100.0      # 初始价格
MU = 0.10       # 年化漂移率（趋势）：0.10 = 平均每年涨 10%
SIGMA = 0.20    # 年化波动率：0.20 = 典型股票的波动水平
DAYS = 252      # 模拟交易日数（一年约 252 个交易日）
SEED = 42       # 随机种子：固定它能复现同一条路径，换数字=换一个平行世界
# ----------------------------------------------


def simulate_gbm(s0, mu, sigma, days, seed):
    """用几何布朗运动模拟一条价格路径。

    核心思想：每天的『对数收益率』是一个正态分布的随机数。
    把每天的随机涨跌累加起来，再取指数，就得到价格路径。
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252                      # 一天 = 1/252 年
    daily_drift = (mu - 0.5 * sigma**2) * dt
    daily_vol = sigma * np.sqrt(dt)
    # 每日对数收益率 ~ 正态(均值=daily_drift, 标准差=daily_vol)
    log_returns = daily_drift + daily_vol * rng.standard_normal(days)
    log_price = np.log(s0) + np.cumsum(log_returns)
    price = np.exp(log_price)
    return price


def main():
    # === 1. 造一段「股价」 ===
    price = simulate_gbm(S0, MU, SIGMA, DAYS, SEED)
    # 用 pandas 的 Series 装起来，带一个「交易日」索引——这就是量化数据的标准形态
    dates = pd.bdate_range("2024-01-01", periods=DAYS)  # 连续工作日
    price = pd.Series(price, index=dates, name="价格")

    # === 2. 算收益率 ===
    # 对数收益率 = ln(今天价格 / 昨天价格)。为什么用 log？因为它可加、更接近正态。
    log_ret = np.log(price / price.shift(1)).dropna()

    # === 3. 算关键指标 ===
    ann_return = log_ret.mean() * 252               # 年化收益（对数）
    ann_vol = log_ret.std() * np.sqrt(252)          # 年化波动率
    sharpe = ann_return / ann_vol                   # 夏普比率（这里假设无风险利率=0）
    total_return = price.iloc[-1] / price.iloc[0] - 1  # 区间总收益
    max_drawdown = ((price / price.cummax()) - 1).min()  # 最大回撤（最痛的一次下跌）

    # === 打印结果（终端里看） ===
    print("=" * 46)
    print("  你的第一份量化「体检报告」")
    print("=" * 46)
    print(f"  模拟参数：年化趋势 {MU:.0%}，年化波动 {SIGMA:.0%}，{DAYS} 个交易日")
    print("-" * 46)
    print(f"  期末价格      : {price.iloc[-1]:8.2f}  (期初 {S0:.2f})")
    print(f"  区间总收益    : {total_return:8.2%}")
    print(f"  年化收益(对数): {ann_return:8.2%}")
    print(f"  年化波动率    : {ann_vol:8.2%}")
    print(f"  夏普比率      : {sharpe:8.2f}   <- 每单位风险换来多少收益")
    print(f"  最大回撤      : {max_drawdown:8.2%}   <- 期间最惨的一次下跌")
    print("=" * 46)
    print("  提示：换个 SEED 重跑，指标会大变——这说明单条路径有很大")
    print("  随机性。别被一次漂亮的结果骗了，这正是『信号 vs 噪声』。")
    print("=" * 46)

    # === 画图并存成文件 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：价格路径
    axes[0].plot(price.index, price.values, color="#1f77b4")
    axes[0].set_title("Simulated Price Path (GBM)")
    axes[0].set_xlabel("date")
    axes[0].set_ylabel("price")
    axes[0].grid(alpha=0.3)

    # 右图：每日收益率的分布（直方图）—— 看看它像不像「钟形」正态
    axes[1].hist(log_ret.values, bins=30, color="#ff7f0e", edgecolor="white")
    axes[1].set_title("Daily Log-Returns Distribution")
    axes[1].set_xlabel("daily log return")
    axes[1].set_ylabel("frequency")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = "第一课配图.png"
    fig.savefig(out_path, dpi=110)
    print(f"  图已保存到: {out_path}  （打开看看价格曲线和收益分布）")


if __name__ == "__main__":
    main()
