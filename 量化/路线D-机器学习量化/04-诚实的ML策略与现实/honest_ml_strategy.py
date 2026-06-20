"""
D4 · 一个诚实的 ML 策略 + 现实（路线 D 收尾）
============================================
把 D1–D3 串起来：用技术特征预测涨跌、做多空策略。
关键是对比【作弊版】(样本内/打乱) 和【诚实版】(走向前验证+交易成本)。
结论往往很扎心——这正是要让你亲眼看到的。

  实验①  净值对比：样本内(光鲜) vs 走向前(诚实) vs 买入持有
  实验②  方向准确率：样本内看着能预测，走向前≈50%(掷硬币)
  实验③  指标对比：样本内夏普 vs 走向前(毛/扣成本) vs 买入持有
  实验④  元过拟合：试 50 个随机配置，走向前夏普散布在 0 附近(最好的=运气)

运行：python honest_ml_strategy.py
依赖：numpy, pandas, scikit-learn, matplotlib（数据用 量化/数据/ 示例行情）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 量化/ 入路径

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import quant_tools as qt

TD = 252
COST = 5/1e4   # 单边 5bp


def make_features(close):
    """构造只用『过去信息』的技术特征 + 标签(明天涨跌)。"""
    r = np.log(close/close.shift(1))
    df = pd.DataFrame(index=close.index)
    df["mom5"] = r.rolling(5).sum()
    df["mom20"] = r.rolling(20).sum()
    df["vol20"] = r.rolling(20).std()
    df["ma_ratio"] = close/close.rolling(20).mean() - 1
    df["rng"] = r.rolling(10).max() - r.rolling(10).min()
    df["ret_next"] = r.shift(-1)              # 明天的收益(标签来源)
    df["y"] = (df["ret_next"] > 0).astype(int)
    return df.dropna()


def strat_metrics(pos, ret_next):
    """给定持仓(0/1)和次日收益，扣成本算策略净收益与夏普。"""
    pos = np.asarray(pos, float)
    turn = np.abs(np.diff(np.concatenate([[0], pos])))
    net = pos*ret_next - turn*COST
    sharpe = net.mean()/net.std()*np.sqrt(TD) if net.std() > 0 else 0.0
    return net, sharpe


def walk_forward(feat, seed=0, cols=None):
    """走向前验证：每次只用过去训练，预测未来一段，滚动前进。"""
    cols = cols or ["mom5", "mom20", "vol20", "ma_ratio", "rng"]
    X = feat[cols].values; y = feat["y"].values; rn = feat["ret_next"].values
    n = len(feat); start = 300; step = 40
    pos = np.zeros(n)
    for t in range(start, n, step):
        model = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=seed, n_jobs=1)
        model.fit(X[:t], y[:t])                      # 只用 t 之前
        end = min(t+step, n)
        pos[t:end] = model.predict(X[t:end])         # 预测未来
    net, sharpe = strat_metrics(pos[start:], rn[start:])
    acc = (pos[start:] == y[start:]).mean()
    return pos[start:], net, sharpe, acc, rn[start:]


def main():
    print("=" * 60)
    print("  D4 · 一个诚实的 ML 策略 + 现实")
    print("=" * 60)
    close = qt.load_sample_data()["Close"]
    feat = make_features(close)
    cols = ["mom5", "mom20", "vol20", "ma_ratio", "rng"]
    X, y, rn = feat[cols].values, feat["y"].values, feat["ret_next"].values

    # 作弊版：全样本训练，再在全样本上预测(用了未来 + 过拟合)
    cheat = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0).fit(X, y)
    pos_cheat = cheat.predict(X)
    net_cheat, sh_cheat = strat_metrics(pos_cheat, rn)
    acc_cheat = (pos_cheat == y).mean()

    # 诚实版：走向前
    pos_wf, net_wf, sh_wf, acc_wf, rn_wf = walk_forward(feat)
    # 走向前『毛收益』(不扣成本)对照
    sh_wf_gross = (pos_wf*rn_wf).mean()/(pos_wf*rn_wf).std()*np.sqrt(TD)
    # 买入持有
    sh_bh = rn_wf.mean()/rn_wf.std()*np.sqrt(TD)

    print(f"① 方向准确率：作弊版 {acc_cheat:.1%} vs 走向前 {acc_wf:.1%}（≈50%=掷硬币）")
    print(f"③ 年化夏普：作弊版 {sh_cheat:.2f} | 走向前(毛) {sh_wf_gross:.2f} | 走向前(扣成本) {sh_wf:.2f} | 买入持有 {sh_bh:.2f}")
    print("   作弊版光鲜亮丽，诚实的走向前+成本后，优势基本蒸发——这是 ML 量化的常态。\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 净值对比
    ax = axes[0, 0]
    idx = feat.index[300:]
    ax.plot(idx, np.exp(np.cumsum(net_cheat[300:])), color="#d62728", label=f"in-sample (cheat) Sh={sh_cheat:.1f}")
    ax.plot(idx, np.exp(np.cumsum(net_wf)), color="#1f77b4", label=f"walk-forward (honest) Sh={sh_wf:.1f}")
    ax.plot(idx, np.exp(np.cumsum(rn_wf)), color="#888", label=f"buy & hold Sh={sh_bh:.1f}")
    ax.set_title("(1) Equity: cheat looks great, honest doesn't")
    ax.set_ylabel("net value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 方向准确率
    ax = axes[0, 1]
    ax.bar(["in-sample\n(cheat)", "walk-forward\n(honest)"], [acc_cheat, acc_wf], color=["#d62728", "#1f77b4"])
    ax.axhline(0.5, color="k", ls="--", label="coin flip 50%")
    ax.set_ylim(0.4, 0.7); ax.set_title("(2) Directional accuracy"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # 图③ 夏普对比
    ax = axes[1, 0]
    names = ["cheat", "WF gross", "WF net", "buy&hold"]
    vals = [sh_cheat, sh_wf_gross, sh_wf, sh_bh]
    ax.bar(names, vals, color=["#d62728", "#ff7f0e", "#1f77b4", "#888"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(3) Sharpe: edge evaporates after honesty"); ax.set_ylabel("annual Sharpe")
    ax.grid(alpha=0.3, axis="y")

    # 图④ 元过拟合：50 个随机配置的走向前夏普
    ax = axes[1, 1]
    sharps = []
    for s in range(24):
        subset = list(np.array(cols)[np.random.default_rng(s).choice(len(cols), size=3, replace=False)])
        _, _, sh, _, _ = walk_forward(feat, seed=s, cols=subset)
        sharps.append(sh)
    sharps = np.array(sharps)
    ax.hist(sharps, bins=15, color="#9467bd", edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(sharps.max(), color="red", ls="--", label=f"best (luck) {sharps.max():.2f}")
    ax.axvline(sh_bh, color="green", ls="--", label=f"buy&hold {sh_bh:.2f}")
    ax.set_title(f"(4) Meta-overfit: {len(sharps)} configs, mean Sharpe {sharps.mean():.2f}")
    ax.set_xlabel("walk-forward Sharpe"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "D4配图.png"
    fig.savefig(out, dpi=110)
    print(f"④ 元过拟合：试 {len(sharps)} 个随机特征配置，走向前夏普均值 {sharps.mean():.2f}，散布在 0 附近。")
    print(f"   最好的 {sharps.max():.2f} 多半是运气；若你只报告它，就是在自欺(紧缩夏普该把它打回去)。\n")
    print(f"  图已保存: {out.name}（净值/准确率/夏普/元过拟合 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
