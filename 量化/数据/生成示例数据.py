"""
生成教学用「示例行情数据」
===========================
为什么要自己造数据？
  - 取真实行情常受网络/接口限制（如本项目的云环境）。
  - 自造数据可复现、可控、随处能跑，最适合教学。

这份数据被刻意设计得「有一点点可被均线策略捕捉的趋势」+「大量噪声」，
好让我们在第 04 课看到策略「毛收益不错」，再到第 05 课用交易成本和
样本外检验把它「打回原形」——这正是真实量化最常见的剧情。

⚠️ 诚实声明：这是【模拟数据】，不是真实股价。真实高效市场上，
   这么简单的策略扣掉成本后通常无效。它只用于学习流程，不代表能赚钱。

运行：python 生成示例数据.py   → 生成 示例数据/示例股_日线.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 101
N_YEARS = 6
TRADING_DAYS = 252
OUT_DIR = Path(__file__).resolve().parent / "示例数据"


def drift_schedule(n):
    """把 6 年分成 6 个阶段，给每段一个『年化漂移率』，模拟市场的牛熊轮回：
    牛市 → 强牛 → 熊市(大跌) → 修复 → 震荡 → 温和牛。
    再做平滑，避免阶段之间生硬跳变。
    """
    annual_drift = [0.22, 0.30, -0.45, 0.28, 0.02, 0.16]
    segments = np.array_split(np.arange(n), len(annual_drift))
    mu = np.zeros(n)
    for d, idx in zip(annual_drift, segments):
        mu[idx] = d
    return pd.Series(mu).rolling(20, min_periods=1).mean().values


def generate_close(n, seed):
    """生成带『阶段性趋势 + 波动聚集 + 厚尾』的对数收益，再累成收盘价。"""
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS

    # 1) 趋势项：分阶段的牛熊漂移（见 drift_schedule）
    mu = drift_schedule(n)

    # 2) 波动聚集：简化版 GARCH，让『大波动后面常跟大波动』
    omega, alpha, beta = 2e-6, 0.08, 0.90
    sigma2 = np.full(n, 0.20**2 * dt)   # 初始日方差（年化~20%）
    shocks = rng.standard_t(df=5, size=n) / np.sqrt(5/3)  # 厚尾(t分布)，标准化到方差1
    ret = np.zeros(n)
    for t in range(1, n):
        sigma2[t] = omega + alpha * ret[t-1]**2 + beta * sigma2[t-1]
        ret[t] = mu[t] * dt + np.sqrt(sigma2[t]) * shocks[t]

    log_price = np.log(100.0) + np.cumsum(ret)
    return np.exp(log_price)


def synth_ohlcv(close, seed):
    """从收盘价合成出近似的 开/高/低/成交量（教学用，让你认识 OHLCV 各列）。"""
    rng = np.random.default_rng(seed + 1)
    n = len(close)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    # 开盘 = 昨收 + 跳空噪声
    open_ = prev_close * (1 + 0.003 * rng.standard_normal(n))
    hi_lo_noise = np.abs(0.006 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + hi_lo_noise)
    low = np.minimum(open_, close) * (1 - hi_lo_noise)
    # 成交量：与当日波动正相关 + 对数正态噪声
    daily_ret = np.abs(np.concatenate([[0], np.diff(np.log(close))]))
    volume = (1e6 * (1 + 20 * daily_ret) * np.exp(0.3 * rng.standard_normal(n))).astype(int)
    return open_, high, low, volume


def main():
    n = N_YEARS * TRADING_DAYS
    close = generate_close(n, SEED)
    open_, high, low, volume = synth_ohlcv(close, SEED)

    dates = pd.bdate_range("2018-01-02", periods=n)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    ).round({"Open": 2, "High": 2, "Low": 2, "Close": 2})
    df.index.name = "Date"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "示例股_日线.csv"
    df.to_csv(out_path, encoding="utf-8")

    print(f"已生成 {len(df)} 行数据 -> {out_path}")
    print(f"日期范围: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"收盘价区间: {df['Close'].min():.2f} ~ {df['Close'].max():.2f}")
    print(df.head(3))


if __name__ == "__main__":
    main()
