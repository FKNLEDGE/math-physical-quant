"""
量化主线 · 共享工具箱
======================
02–05 课复用的函数都放这里，避免重复造轮子。
设计原则：每个函数只做一件事，名字说人话，注释讲清「为什么」。

约定（很重要）：
  - 用【对数收益率】做计算（可加、近正态）。
  - 回测一律把信号【右移一天】(shift(1))，杜绝『前视偏差』——
    今天收盘才算出的信号，最早明天才能交易。
"""

from pathlib import Path
import numpy as np
import pandas as pd

# 一年的交易日数（年化换算用）
TRADING_DAYS = 252


# ============ 一、数据 ============

def sample_data_path() -> Path:
    """稳健地定位示例数据 CSV（无论从哪个目录运行都能找到）。"""
    return Path(__file__).resolve().parent / "数据" / "示例数据" / "示例股_日线.csv"


def load_sample_data() -> pd.DataFrame:
    """载入教学用示例行情（OHLCV，DatetimeIndex）。"""
    path = sample_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"找不到示例数据：{path}\n请先到 量化/数据/ 运行：python 生成示例数据.py"
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def fetch_real_data(symbol="AAPL", start="2018-01-01", end=None, source="yfinance"):
    """在你自己的机器上获取真实行情（云环境可能被网络限制）。

    source="yfinance" 取海外（美股等）；source="akshare" 取 A 股。
    取不到时给出友好提示，并建议回退到 load_sample_data()。
    """
    try:
        if source == "yfinance":
            import yfinance as yf
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):       # 新版 yfinance 可能多层列
                df.columns = df.columns.get_level_values(0)
            return df
        elif source == "akshare":
            import akshare as ak  # 需先 pip install akshare
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start.replace("-", ""),
                                    end_date=(end or "20991231").replace("-", ""),
                                    adjust="qfq")
            df = df.rename(columns={"日期": "Date", "开盘": "Open", "最高": "High",
                                    "最低": "Low", "收盘": "Close", "成交量": "Volume"})
            return df.set_index(pd.to_datetime(df["Date"]))[["Open", "High", "Low", "Close", "Volume"]]
        else:
            raise ValueError("source 只支持 'yfinance' 或 'akshare'")
    except Exception as e:
        raise RuntimeError(
            f"取数失败（可能是网络/接口限制）：{e}\n"
            f"建议先用 load_sample_data() 的示例数据学习流程。"
        ) from e


# ============ 二、收益率 ============

def to_log_returns(close: pd.Series) -> pd.Series:
    """收盘价 -> 每日对数收益率。"""
    return np.log(close / close.shift(1)).dropna()


def to_equity_curve(returns: pd.Series, start_value=1.0) -> pd.Series:
    """收益率序列 -> 累计净值曲线（从 start_value 开始）。"""
    return start_value * np.exp(returns.cumsum())


# ============ 三、绩效指标 ============

def max_drawdown(equity: pd.Series) -> float:
    """最大回撤：从历史最高点跌下来的最惨幅度（负数）。"""
    return ((equity / equity.cummax()) - 1).min()


def perf_metrics(returns: pd.Series, freq=TRADING_DAYS) -> dict:
    """常用绩效指标打包。returns 为【对数】日收益率。"""
    equity = to_equity_curve(returns)
    ann_return = returns.mean() * freq
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    return {
        "总收益": float(np.exp(returns.sum()) - 1),
        "年化收益": float(ann_return),
        "年化波动": float(ann_vol),
        "夏普比率": float(sharpe),
        "最大回撤": float(max_drawdown(equity)),
        "胜率": float((returns > 0).mean()),
    }


def print_metrics(name: str, m: dict):
    """整齐地打印一份绩效指标。"""
    print(f"  [{name}]")
    print(f"    总收益 {m['总收益']:7.2%} | 年化 {m['年化收益']:7.2%} | "
          f"波动 {m['年化波动']:6.2%} | 夏普 {m['夏普比率']:5.2f} | "
          f"最大回撤 {m['最大回撤']:7.2%} | 胜率 {m['胜率']:5.1%}")


# ============ 四、均线交叉策略 + 回测 ============

def ma_crossover_position(close: pd.Series, fast=20, slow=60) -> pd.Series:
    """均线交叉的【持仓信号】：快线在慢线上方→满仓(1)，否则空仓(0)。

    关键：信号右移一天(shift(1))，因为今天收盘才知道均线关系，
    最早明天开盘才能据此交易——这样杜绝前视偏差。
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    raw = (fast_ma > slow_ma).astype(float)
    return raw.shift(1).fillna(0.0)


def backtest_long_only(close: pd.Series, position: pd.Series, cost_bps=0.0) -> pd.DataFrame:
    """对『只做多/空仓』策略做向量化回测。

    cost_bps：单边交易成本（基点，1bp=0.01%）。换手时按换手量扣成本。
    返回含 持仓、市场收益、策略收益、净值 的 DataFrame。
    """
    mkt_ret = to_log_returns(close)
    position = position.reindex(mkt_ret.index).fillna(0.0)

    # 持仓变化 -> 换手 -> 成本（每次买入或卖出都付一次单边成本）
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 1e4)

    strat_ret = position * mkt_ret - cost
    out = pd.DataFrame({
        "持仓": position,
        "市场收益": mkt_ret,
        "策略收益": strat_ret,
        "市场净值": to_equity_curve(mkt_ret),
        "策略净值": to_equity_curve(strat_ret),
    })
    return out
