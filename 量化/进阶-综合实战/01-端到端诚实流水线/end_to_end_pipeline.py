"""
进阶·综合实战 ① 端到端「诚实流水线」
========================================
把整个项目的进阶武器,在一条流水线里串起来,跑一遍真实研究该有的样子:

  数据 → 特征(技术 + 分数阶差分) → 三重栅栏标签 → 元标注(主信号+次模型)
       → Purged 走向前验证 → 计入执行成本 → 诚实评估(夏普 + 稳健性)

  核心看点是【诚实瀑布】:每加一层"别自欺",纸面收益就缩水一截,
  最后剩下的才是你真能拿到的。这把 D4 / 诚实三件套 / 微观执行 收束成一张图。

  实验①  净值: 样本内(光鲜) vs 端到端诚实 vs 买入持有
  实验②  三重栅栏标签 + 元标注过滤后的下注比例
  实验③  诚实瀑布: 夏普如何随"样本内→走向前→Purged→计成本"层层缩水
  实验④  稳健性: 多个随机特征子集的诚实夏普分布(像不像运气?)

运行：python end_to_end_pipeline.py
依赖：numpy, pandas, scikit-learn, statsmodels, matplotlib（数据用 量化/数据/ 示例行情）
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
COST = 8 / 1e4           # 单边执行成本(价差/2+冲击的简化), 8bp
PT, SL, VERT = 1.5, 0.7, 20
FEATS = ["mom5", "mom20", "vol20", "ma_ratio", "rng", "ffd"]


def ffd_weights(d, thresh=1e-4, max_k=2000):
    w = [1.0]
    for k in range(1, max_k):
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
    return np.array(w[::-1])


def frac_diff(series, d=0.4, thresh=1e-4):
    w = ffd_weights(d, thresh); width = len(w)
    v = series.values.astype(float); out = np.full(len(v), np.nan)
    if width <= len(v):
        out[width - 1:] = np.correlate(v, w, mode="valid")
    return pd.Series(out, index=series.index)


def build_events(close):
    """技术特征(+分数阶差分) → 动量主信号处建事件 → 三重栅栏元标注。"""
    r = np.log(close / close.shift(1))
    f = pd.DataFrame(index=close.index)
    f["mom5"] = r.rolling(5).sum(); f["mom20"] = r.rolling(20).sum()
    f["vol20"] = r.rolling(20).std(); f["ma_ratio"] = close / close.rolling(20).mean() - 1
    f["rng"] = r.rolling(10).max() - r.rolling(10).min()
    f["ffd"] = frac_diff(np.log(close), 0.4)              # 分数阶差分特征(平稳+留记忆)
    sigma = r.rolling(20).std(); logp = np.log(close.values); n = len(close)
    rows = []
    for i in range(60, n - VERT):
        if not np.isfinite(f["mom20"].iloc[i]) or f["mom20"].iloc[i] <= 0:
            continue
        si = sigma.iloc[i]
        if not np.isfinite(si) or si <= 0 or not np.isfinite(f["ffd"].iloc[i]):
            continue
        up, dn = PT * si * np.sqrt(VERT), SL * si * np.sqrt(VERT)
        p0 = logp[i]; hit = VERT
        for j in range(1, VERT + 1):
            g = logp[i + j] - p0
            if g >= up or g <= -dn:
                hit = j; break
        exret = logp[i + hit] - p0
        rows.append({"i": i, "hit": hit, "exret": exret, "meta": int(exret > 0),
                     **{c: f[c].iloc[i] for c in FEATS}})
    return pd.DataFrame(rows)


def daily_pnl(ev, take, bar0, bar1, with_cost):
    """flat-gating 单仓执行,把每笔收益均摊到持有日;返回日 P&L 序列。"""
    L = bar1 - bar0; d = np.zeros(L); nf = bar0 - 1
    for k in range(len(ev)):
        i = int(ev["i"].iat[k])
        if i <= nf or not take[k]:
            continue
        hit = int(ev["hit"].iat[k]); net = ev["exret"].iat[k] - (2 * COST if with_cost else 0)
        a = i - bar0; b = min(a + hit, L)
        if a < 0:
            continue
        d[a:b] += net / max(hit, 1); nf = i + hit
    return d


def sharpe(d):
    return d.mean() / d.std() * np.sqrt(TD) if d.std() > 0 else 0.0


def wf_proba(ev, cols, purge, start_frac=0.4, step=40, embargo=VERT):
    """走向前训练次模型,可选 Purge+Embargo 清洗重叠样本。返回每事件赢面概率。"""
    X = ev[cols].values; y = ev["meta"].values
    t0 = ev["i"].values; t1 = ev["i"].values + ev["hit"].values
    n = len(ev); start = int(n * start_frac); proba = np.full(n, np.nan)
    for t in range(start, n, step):
        tr = np.ones(t, bool)
        if purge:
            a, b = t0[t:min(t + step, n)].min(), t1[t:min(t + step, n)].max()
            # 清洗:训练集中标签窗口与测试块重叠者剔除(这里测试块在末端,清洗其前缘)
            tr &= ~(~((t1[:t] < a) | (t0[:t] > b + embargo)))
        if tr.sum() < 50:
            continue
        rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=25,
                                    random_state=0, n_jobs=1).fit(X[:t][tr], y[:t][tr])
        end = min(t + step, n); proba[t:end] = rf.predict_proba(X[t:end])[:, 1]
    return proba, start


def main():
    print("=" * 64)
    print("  进阶·综合实战 ① 端到端「诚实流水线」")
    print("=" * 64)
    close = qt.load_sample_data()["Close"]
    ev = build_events(close); n = len(close)
    print(f"  事件 {len(ev)} 个(动量主信号+三重栅栏); 特征含分数阶差分; 执行成本 {COST*1e4:.0f}bp/单边\n")

    # —— 诚实瀑布:四个阶段的夏普 ——
    # 阶段1 样本内作弊:全样本训练RF,再在全样本预测,无成本
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=25,
                                random_state=0, n_jobs=1).fit(ev[FEATS], ev["meta"])
    p_in = rf.predict_proba(ev[FEATS])[:, 1]
    take_in = p_in >= 0.5
    bar0_all = int(ev["i"].iloc[0])
    sh1 = sharpe(daily_pnl(ev, take_in, bar0_all, n, with_cost=False))

    # 阶段2 走向前(无purge,无成本)
    pf, start = wf_proba(ev, FEATS, purge=False)
    te = ev.iloc[start:].reset_index(drop=True); pf_te = pf[start:]; bar0 = int(te["i"].iloc[0])
    take2 = pf_te >= 0.5
    sh2 = sharpe(daily_pnl(te, take2, bar0, n, with_cost=False))

    # 阶段3 走向前+Purge+Embargo(无成本)
    pp, _ = wf_proba(ev, FEATS, purge=True)
    pp_te = pp[start:]; take3 = pp_te >= 0.5
    sh3 = sharpe(daily_pnl(te, take3, bar0, n, with_cost=False))

    # 阶段4 +执行成本
    d4 = daily_pnl(te, take3, bar0, n, with_cost=True); sh4 = sharpe(d4)

    # 基准:买入持有(测试段)
    mkt = np.log(close / close.shift(1)).values[bar0:n]
    sh_bh = np.nanmean(mkt) / np.nanstd(mkt) * np.sqrt(TD)

    stages = ["in-sample\n(cheat)", "walk-fwd", "+purge\n+embargo", "+exec cost\n(HONEST)"]
    shs = [sh1, sh2, sh3, sh4]
    print("③ 诚实瀑布(夏普逐层缩水):")
    for s, v in zip(stages, shs):
        print(f"   {s.replace(chr(10),' '):24s} {v:+.2f}")
    print(f"   买入持有基准: {sh_bh:+.2f}")
    print(f"   从样本内 {sh1:+.2f} 一路诚实到 {sh4:+.2f}——纸面与现实的差距,全是自欺挤出来的水分。\n")

    # —— 元标注过滤比例 ——
    print(f"② 三重栅栏: 止盈 {(ev['exret']>0).mean():.0%} 赚 / 亏 {(ev['exret']<=0).mean():.0%};"
          f" 元标注(诚实阶段)采纳 {take3.mean():.0%} 的信号\n")

    # —— 稳健性:随机特征子集的诚实夏普分布 ——
    rng = np.random.default_rng(0); robust = []
    for s in range(30):
        sub = list(rng.choice(FEATS, size=4, replace=False))
        psub, _ = wf_proba(ev, sub, purge=True)
        tk = psub[start:] >= 0.5
        robust.append(sharpe(daily_pnl(te, tk, bar0, n, with_cost=True)))
    robust = np.array(robust)
    print(f"④ 稳健性: 30 个随机特征子集的诚实夏普 均值 {robust.mean():+.2f} 标准差 {robust.std():.2f};"
          f" 正比例 {(robust>0).mean():.0%}\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    idx = close.index[bar0:n]

    ax = axes[0, 0]
    eq_in = np.exp(np.cumsum(daily_pnl(ev, take_in, bar0_all, n, False)))[(bar0 - bar0_all):]
    eq_h = np.exp(np.cumsum(d4))
    eq_bh = np.exp(np.nancumsum(np.nan_to_num(mkt)))
    ax.plot(idx, eq_in[:len(idx)], color="#d62728", label=f"in-sample (cheat) Sh={sh1:.2f}")
    ax.plot(idx, eq_h, color="#1f77b4", label=f"end-to-end HONEST Sh={sh4:.2f}")
    ax.plot(idx, eq_bh, color="#888", label=f"buy & hold Sh={sh_bh:.2f}")
    ax.set_title("(1) Equity: in-sample dazzles, honest is sober")
    ax.set_ylabel("net value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    parts = [(ev["exret"] > 0).mean(), (ev["exret"] <= 0).mean()]
    ax.bar(["win", "lose"], parts, color=["#2ca02c", "#d62728"])
    ax.axhline(take3.mean(), color="#1f77b4", ls="--", lw=1.5, label=f"meta accepts {take3.mean():.0%}")
    ax.set_title("(2) Triple-barrier outcomes + meta-label acceptance")
    ax.set_ylabel("fraction"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    colors = ["#d62728", "#ff7f0e", "#ffbb33", "#2ca02c"]
    bars = ax.bar(stages, shs, color=colors)
    ax.axhline(sh_bh, color="#888", ls="--", lw=1.5, label=f"buy & hold {sh_bh:.2f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(3) Honesty waterfall: Sharpe shrinks as self-deception is removed")
    ax.set_ylabel("annualized Sharpe"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, shs):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08), f"{v:.2f}", ha="center", fontsize=9)

    ax = axes[1, 1]
    ax.hist(robust, bins=12, color="#9467bd", edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(robust.mean(), color="#d62728", ls="--", lw=1.5, label=f"mean {robust.mean():.2f}")
    ax.axvline(sh_bh, color="#2ca02c", ls="--", lw=1.5, label=f"buy&hold {sh_bh:.2f}")
    ax.set_title("(4) Robustness: honest Sharpe over 30 random feature subsets")
    ax.set_xlabel("honest Sharpe"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "综合实战配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（净值/标签与元标注/诚实瀑布/稳健性 四合一）")
    print("=" * 64)


if __name__ == "__main__":
    main()
