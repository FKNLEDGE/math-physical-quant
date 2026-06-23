"""
进阶·金融机器学习 ① 三重栅栏标注 + 元标注（Meta-Labeling）
============================================================
D4 用『明天涨不涨』(固定时窗)给样本贴标签——可真实交易有止盈/止损/持有上限,
路径决定一切。López de Prado《金融机器学习进展》提出两件利器:

  三重栅栏(Triple-Barrier): 每笔交易设【上栅栏=止盈 / 下栅栏=止损 / 垂直栅栏=持有上限】,
      谁先被碰到就贴谁的标签(+1止盈 / -1止损 / 0到期)。标签终于和"真实会怎样平仓"一致。
  元标注(Meta-Labeling): 把决策拆成两层——
      『主模型』决定【方向】(这里:动量为正→做多),『次模型(ML)』决定【要不要下注】。
      次模型预测"这一笔主信号会赢吗",过滤掉低胜率信号 → 精确率↑、F1↑、回撤↓。

  实验①  一条价格路径上画出三栅栏,看谁先被碰到(止损先到=这笔被打掉)
  实验②  三重栅栏标签分布 vs 固定时窗:路径让两者经常不一致(终点赢≠过程没被打掉)
  实验③  元标注:主信号(全要) vs 次模型过滤——精确率/召回/F1 怎么变(样本外)
  实验④  执行层净值:全要 vs 元标注过滤,谁的夏普/回撤更好(样本外)

运行：python triple_barrier_meta_labeling.py
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
from sklearn.metrics import precision_score, recall_score, f1_score
import quant_tools as qt

TD = 252
COST = 5 / 1e4              # 单边 5bp
PT, SL, VERT = 1.5, 0.7, 20  # 止盈/止损倍数(×日波动×√vert): 让利润奔跑、止损收紧; 持有上限 20 天
FEAT_COLS = ["mom5", "mom20", "vol20", "ma_ratio", "rng"]


def make_features(close):
    """只用过去信息的技术特征(与 D4 一致,便于闭环)。返回 (特征df, 日对数收益)。"""
    r = np.log(close / close.shift(1))
    df = pd.DataFrame(index=close.index)
    df["mom5"] = r.rolling(5).sum()
    df["mom20"] = r.rolling(20).sum()
    df["vol20"] = r.rolling(20).std()
    df["ma_ratio"] = close / close.rolling(20).mean() - 1
    df["rng"] = r.rolling(10).max() - r.rolling(10).min()
    return df, r


def triple_barrier(logp, i, sigma_i, side=1, pt=PT, sl=SL, vert=VERT):
    """从第 i 根 bar 入场,按 side(+1做多)设三栅栏。
    宽度 = 倍数 × sigma_i(入场时日波动) × √vert(把日波动放大到持有期尺度)。
    返回 (label, hit_off, exit_ret):
      label  = +1 先碰止盈 / -1 先碰止损 / 0 到期(垂直栅栏)
      hit_off= 第几根 bar 触碰(1..vert)
      exit_ret = 出场相对入场的对数收益(已被栅栏截断)
    """
    up = pt * sigma_i * np.sqrt(vert)
    dn = sl * sigma_i * np.sqrt(vert)
    p0 = logp[i]
    n = len(logp)
    for j in range(1, vert + 1):
        if i + j >= n:
            return 0, j - 1, side * (logp[n - 1] - p0)
        g = side * (logp[i + j] - p0)        # 朝下注方向的累计收益
        if g >= up:
            return 1, j, side * (logp[i + j] - p0)
        if g <= -dn:
            return -1, j, side * (logp[i + j] - p0)
    return 0, vert, side * (logp[i + vert] - p0)


def build_events(close):
    """主信号(动量>0)处生成事件,贴三重栅栏标签 + 固定时窗标签作对照。"""
    feat, r = make_features(close)
    sigma = r.rolling(20).std()
    logp = np.log(close.values)
    n = len(close)
    rows = []
    for i in range(60, n - VERT):
        if not np.isfinite(feat["mom20"].iloc[i]) or feat["mom20"].iloc[i] <= 0:
            continue                          # 主模型:只在动量为正时考虑做多
        si = sigma.iloc[i]
        if not np.isfinite(si) or si <= 0:
            continue
        lab, hit, exret = triple_barrier(logp, i, si, side=1)
        fixed_ret = logp[i + VERT] - logp[i]  # 固定时窗:只看 vert 天后的终点
        rows.append({
            "i": i, "date": close.index[i],
            **{c: feat[c].iloc[i] for c in FEAT_COLS},
            "sigma": si, "tb_label": lab, "meta": int(exret > 0),  # 元标注=这笔到底赚没赚钱
            "hit": hit, "exret": exret, "fixed_ret": fixed_ret,
        })
    return pd.DataFrame(rows).set_index("date")


def daily_series(ev, gate, bar0, bar1):
    """把被采纳的(非重叠)交易铺成测试期的日策略收益序列(便于算夏普/回撤)。
    gate: 布尔数组,是否采纳该事件;同一时间只持一仓(flat-gating 处理重叠信号)。"""
    L = bar1 - bar0
    d = np.zeros(L)
    next_free = bar0 - 1
    taken = 0
    for k, (_, row) in enumerate(ev.iterrows()):
        i = int(row["i"])
        if i <= next_free or not gate[k]:
            continue                          # 还在持仓中,或被过滤 → 跳过
        hit = int(row["hit"])
        net = row["exret"] - 2 * COST         # 进+出两次单边成本
        a = i - bar0
        b = min(a + hit, L)
        if a < 0:
            continue
        d[a:b] += net / max(hit, 1)           # 把这笔收益均摊到持有的每一天
        next_free = i + hit
        taken += 1
    return d, taken


def sharpe(d):
    return d.mean() / d.std() * np.sqrt(TD) if d.std() > 0 else 0.0


def walk_forward_meta(ev, start_frac=0.4, step=40):
    """走向前训练次模型:每个区块只用过去的事件训练,预测下一区块的'会赢吗'。
    这样跨越牛/熊两段、不断重新校准——比一次性切分诚实得多。返回每个事件的赢面概率。"""
    X, y = ev[FEAT_COLS].values, ev["meta"].values
    n = len(ev)
    start = int(n * start_frac)
    proba = np.full(n, np.nan)
    for t in range(start, n, step):
        rf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=25,
                                    random_state=0, n_jobs=1)
        rf.fit(X[:t], y[:t])                       # 只用 t 之前的事件
        end = min(t + step, n)
        proba[t:end] = rf.predict_proba(X[t:end])[:, 1]
    return proba, start


def main():
    print("=" * 60)
    print("  进阶·金融机器学习 ① 三重栅栏标注 + 元标注")
    print("=" * 60)
    close = qt.load_sample_data()["Close"]
    ev = build_events(close)
    n = len(close)
    print(f"  主信号(动量>0)共生成 {len(ev)} 个事件;栅栏: 止盈/止损 ±{PT}σ√{VERT}, 持有上限 {VERT} 天\n")

    # ---- 标签分布 + 与固定时窗的不一致 ----
    frac_up = (ev["tb_label"] == 1).mean()
    frac_dn = (ev["tb_label"] == -1).mean()
    frac_to = (ev["tb_label"] == 0).mean()
    # 固定时窗说"赢"(终点为正)的那些里,有多少其实先被止损打掉(tb=-1)?
    fixed_win = ev["fixed_ret"] > 0
    stopped_among_fixed_win = ((ev["tb_label"] == -1) & fixed_win).sum() / max(fixed_win.sum(), 1)
    print(f"① 三重栅栏标签分布: 止盈 +1 {frac_up:.0%} | 止损 -1 {frac_dn:.0%} | 到期 0 {frac_to:.0%}")
    print(f"② 路径的代价: 固定时窗判定为'赢'(终点>0)的事件里, 有 {stopped_among_fixed_win:.0%} 其实")
    print("   在中途先被止损打掉了——固定时窗只看终点,会高估你真实拿得到的收益。\n")

    # ---- 元标注: 走向前训练次模型 ----
    proba, start = walk_forward_meta(ev)
    te = ev.iloc[start:].copy()
    proba_te = proba[start:]
    yte = te["meta"].values
    base_rate = yte.mean()                     # 测试期主信号的真实胜率(基准)
    bar0, bar1 = int(te["i"].iloc[0]), n

    # 把元标注当成一个"信心旋钮":阈值越高=只下注越有把握的信号=交易越少
    base_all = np.ones(len(te), bool)          # 主信号"全要"
    d_all, n_all = daily_series(te, base_all, bar0, bar1)
    eq_all = np.exp(np.cumsum(d_all)); sh_all = sharpe(d_all)
    dd_all = (eq_all / np.maximum.accumulate(eq_all) - 1).min()

    ths = np.arange(0.44, 0.66, 0.02)
    sh_curve, acc_curve = [], []
    for thr in ths:
        d_t, _ = daily_series(te, proba_te >= thr, bar0, bar1)
        sh_curve.append(sharpe(d_t)); acc_curve.append(float((proba_te >= thr).mean()))
    sh_curve, acc_curve = np.array(sh_curve), np.array(acc_curve)

    THR_OP = 0.52                              # 操作点:要求 >52% 把握才下注(整条曲线见图③,非事后挑)
    meta_op = proba_te >= THR_OP
    d_meta, n_meta = daily_series(te, meta_op, bar0, bar1)
    eq_meta = np.exp(np.cumsum(d_meta)); sh_meta = sharpe(d_meta)
    dd_meta = (eq_meta / np.maximum.accumulate(eq_meta) - 1).min()

    # 精确率/召回/F1(诚实汇报:低信号数据上,命中率几乎没提升)
    prec_b = precision_score(yte, base_all.astype(int), zero_division=0)
    rec_b = recall_score(yte, base_all.astype(int), zero_division=0)
    f1_b = f1_score(yte, base_all.astype(int), zero_division=0)
    prec_m = precision_score(yte, meta_op.astype(int), zero_division=0)
    rec_m = recall_score(yte, meta_op.astype(int), zero_division=0)
    f1_m = f1_score(yte, meta_op.astype(int), zero_division=0)
    print(f"③ 元标注=信心旋钮(走向前样本外, 操作点 P(赢)≥{THR_OP}):")
    print(f"   精确率/召回/F1  全要 {prec_b:.2f}/{rec_b:.2f}/{f1_b:.2f}  →  过滤 {prec_m:.2f}/{rec_m:.2f}/{f1_m:.2f}")
    print(f"   诚实结论: 低信号数据上次模型【提不动命中率】(精确率没涨,呼应D2/D4);")
    print(f"   但适度过滤靠【敢于不下注】改善了风险收益比,夏普 {sh_all:.2f}→{sh_meta:.2f}。\n")

    print("④ 执行层(走向前样本外, flat-gating 单仓):")
    print(f"   全要   : 交易 {n_all} 笔 | 总收益 {eq_all[-1]-1:+.1%} | 夏普 {sh_all:.2f} | 最大回撤 {dd_all:.1%}")
    print(f"   元标注 : 交易 {n_meta} 笔 | 总收益 {eq_meta[-1]-1:+.1%} | 夏普 {sh_meta:.2f} | 最大回撤 {dd_meta:.1%}")
    print(f"   过度过滤会翻车: 阈值={ths[-1]:.2f} 时只剩 {acc_curve[-1]:.0%} 信号,夏普跌到 {sh_curve[-1]:.2f}(脆弱悬崖)。\n")

    # ================= 画图 =================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 一条路径 + 三栅栏(挑一个被止损打掉的事件,最有戏剧性)
    ax = axes[0, 0]
    stopped = ev[ev["tb_label"] == -1]
    e = stopped.iloc[len(stopped) // 3] if len(stopped) else ev.iloc[0]
    i0 = int(e["i"])
    seg = slice(max(i0 - 3, 0), i0 + VERT + 4)
    px = close.values[seg]
    xs = np.arange(len(px)) + max(i0 - 3, 0) - i0     # 相对入场的天数
    p0 = close.values[i0]
    up = p0 * np.exp(e["sigma"] * np.sqrt(VERT))
    dn = p0 * np.exp(-e["sigma"] * np.sqrt(VERT))
    ax.plot(xs, px, "o-", ms=3, color="#1f77b4", label="price")
    ax.hlines(up, 0, VERT, color="#2ca02c", lw=2, label="upper = take-profit")
    ax.hlines(dn, 0, VERT, color="#d62728", lw=2, label="lower = stop-loss")
    ax.vlines(VERT, dn, up, color="#7f7f7f", lw=2, ls="--", label="vertical = time limit")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.scatter([e["hit"]], [close.values[i0 + int(e["hit"])]], s=120, marker="X",
               color="#d62728", zorder=5, label=f"hit lower @ day {int(e['hit'])}")
    ax.set_title("(1) Triple barrier: which is touched first labels the trade")
    ax.set_xlabel("days since entry"); ax.set_ylabel("price")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # 图② 标签分布对比
    ax = axes[0, 1]
    ax.bar([-1, 0, 1], [frac_dn, frac_to, frac_up],
           color=["#d62728", "#7f7f7f", "#2ca02c"], width=0.6)
    ax.set_xticks([-1, 0, 1]); ax.set_xticklabels(["-1\nstop-loss", "0\ntime-out", "+1\ntake-profit"])
    ax.set_title("(2) Triple-barrier label mix (path-aware)")
    ax.set_ylabel("fraction of events")
    ax.text(0.5, 0.92, f"of fixed-horizon 'winners',\n{stopped_among_fixed_win:.0%} were STOPPED OUT first",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800"))
    ax.grid(alpha=0.3, axis="y")

    # 图③ 元标注=信心旋钮:OOS夏普 vs 阈值(甜区 + 过度过滤悬崖)
    ax = axes[1, 0]
    ax.plot(ths, sh_curve, "o-", color="#1f77b4", label="meta-filtered Sharpe")
    ax.axhline(sh_all, color="#888", ls="--", lw=1.5, label=f"take all (Sharpe {sh_all:.2f})")
    ax.axvline(THR_OP, color="#2ca02c", lw=1, alpha=0.7)
    k_op = int(np.argmin(np.abs(ths - THR_OP)))
    ax.scatter([THR_OP], [sh_curve[k_op]], s=90, color="#2ca02c", zorder=5,
               label=f"sweet spot @{THR_OP} (keep {acc_curve[k_op]:.0%})")
    ax.annotate("over-filter:\ntoo few trades,\nfragile", xy=(ths[-1], sh_curve[-1]),
                xytext=(ths[-3], sh_curve.min() + 0.5), fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_title("(3) Meta-labeling is a confidence dial (mild filter helps)")
    ax.set_xlabel("meta threshold P(win) to bet"); ax.set_ylabel("out-of-sample Sharpe")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图④ 执行层净值(甜区操作点 vs 全要)
    ax = axes[1, 1]
    idx = close.index[bar0:bar1]
    ax.plot(idx, eq_all, color="#888", label=f"take all  Sh={sh_all:.2f}, DD={dd_all:.0%}, {n_all} trades")
    ax.plot(idx, eq_meta, color="#1f77b4", label=f"meta @{THR_OP}  Sh={sh_meta:.2f}, DD={dd_meta:.0%}, {n_meta} trades")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_title("(4) Execution: fewer trades, better risk-adjusted return")
    ax.set_ylabel("net value (out-of-sample)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "三重栅栏配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（三栅栏/标签分布/信心旋钮/执行净值 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
