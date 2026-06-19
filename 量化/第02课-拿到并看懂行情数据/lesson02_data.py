"""
第 02 课 · 拿到并看懂行情数据
==============================
量化的第一步永远是数据。这节课你会：
  1. 把一份行情数据载入成 pandas 的 DataFrame（量化数据的标准形态）。
  2. 学会『四件套』体检：看头尾、看结构、看统计、查缺失。
  3. 认识 OHLCV 五列各是什么、为什么常用『收盘价/复权价』。
  4. 知道在自己电脑上怎么取【真实】数据（yfinance / akshare）。

运行：python lesson02_data.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 把 量化/ 加入路径，便于 import 工具箱

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quant_tools as qt


def inspect(df):
    """对一份行情数据做『四件套』体检。"""
    print("① 看头尾（确认数据长什么样）")
    print(df.head(3))
    print("  ……")
    print(df.tail(3))

    print("\n② 看结构（行数、列名、类型、索引）")
    print(f"  形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  索引: {df.index.name}（类型 {df.index.dtype}）")
    print(f"  列名: {list(df.columns)}")
    print(f"  日期范围: {df.index.min().date()} ~ {df.index.max().date()}")

    print("\n③ 看统计（每列的均值、极值、分位数）")
    print(df[["Close", "Volume"]].describe().round(2))

    print("\n④ 查缺失（量化里缺失值是头号数据坑）")
    miss = df.isna().sum()
    print(f"  各列缺失数: { '  '.join(f'{c}={int(v)}' for c, v in miss.items()) }")
    print("  ✅ 无缺失，可以开工" if miss.sum() == 0 else "  ⚠️ 有缺失，需先处理（删除/前向填充）")


def main():
    print("=" * 56)
    print("  第 02 课 · 拿到并看懂行情数据")
    print("=" * 56)

    # 载入示例数据（任何环境都能跑）
    df = qt.load_sample_data()
    inspect(df)

    # OHLCV 五列说明
    print("\n" + "-" * 56)
    print("  OHLCV 五列在说什么（以某一天为例）")
    print("-" * 56)
    row = df.iloc[100]
    print(f"  Open  开盘价 = {row['Open']:.2f}   （当天第一笔成交价）")
    print(f"  High  最高价 = {row['High']:.2f}   （当天最高）")
    print(f"  Low   最低价 = {row['Low']:.2f}    （当天最低）")
    print(f"  Close 收盘价 = {row['Close']:.2f}   （当天最后一笔，最常用）")
    print(f"  Volume成交量 = {int(row['Volume']):,}（当天成交股数，反映活跃度）")
    print("  💡 做研究通常用【收盘价】，且真实数据要用【复权价】(adjusted)，")
    print("     否则分红/拆股会在价格上造成假跳空。本示例已是干净的连续价。")

    # 画图：价格 + 成交量（量化的第一直觉来自看图）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, df["Close"], color="#1f77b4", lw=1.2)
    ax1.set_title("Sample Stock — Close Price (2018–2023)")
    ax1.set_ylabel("price")
    ax1.grid(alpha=0.3)
    ax2.bar(df.index, df["Volume"], color="#9aa", width=1.0)
    ax2.set_ylabel("volume")
    ax2.set_xlabel("date")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "第02课配图.png"
    fig.savefig(out, dpi=110)
    print(f"\n  图已保存: {out.name}")

    # 如何取真实数据（在你自己的机器上）
    print("\n" + "-" * 56)
    print("  在自己电脑上取【真实】数据（本云环境联网受限，故用示例数据）")
    print("-" * 56)
    print("  海外(美股)：")
    print("    df = qt.fetch_real_data('AAPL', start='2018-01-01', source='yfinance')")
    print("  A股：先 pip install akshare，再：")
    print("    df = qt.fetch_real_data('600519', start='2018-01-01', source='akshare')")
    print("  取不到时自动提示，可回退 qt.load_sample_data() 先学流程。")


if __name__ == "__main__":
    main()
