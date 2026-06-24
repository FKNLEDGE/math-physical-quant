"""
进阶·金融机器学习 ② 分数阶差分 + Purged CV + PBO（诚实三件套）
================================================================
把 D4 的"别自欺"推到极致。López de Prado 的三件武器,各治一种自欺:

  分数阶差分(Fractional Diff): 价格非平稳(有记忆但统计量漂移),一阶差分(收益)平稳
      却把记忆全删了。分数阶差分用 d∈(0,1) 找【最小的 d* 既让序列平稳、又最大限度留住记忆】。
  Purged + Embargo 交叉验证: 三重栅栏的标签持有期会【重叠】,普通交叉验证里训练集会
      "偷看"测试集的未来→分数虚高。清洗(purge)掉重叠样本 + 隔离(embargo)→打回真实水平。
  PBO 回测过拟合概率(CSCV): 你试了很多策略、挑出"样本内最好"的——它样本外多半泯然众人。
      PBO 量化"你这个漂亮回测有多大概率是蒙出来的"。

  实验①  分数阶差分: 平稳性(ADF) vs 记忆(相关性) 随 d 变化,找 d*
  实验②  三条序列: 原始logP(非平稳/有记忆) vs 分数差分d*(平稳+留记忆) vs 一阶差分(平稳/无记忆)
  实验③  CV 泄露: 乱序CV(虚高) → 分块CV → 分块+Purge+Embargo(诚实),AUC 一路打回原形
  实验④  PBO(CSCV): 纯运气策略池挑"样本内最好",样本外排名≈掷硬币 → PBO≈50%

运行：python frac_diff_purged_cv_pbo.py
依赖：numpy, pandas, scikit-learn, statsmodels, matplotlib（数据用 量化/数据/ 示例行情）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 量化/ 入路径

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.stattools import adfuller
import quant_tools as qt

RNG = np.random.default_rng(0)
VERT = 20
FEAT_COLS = ["mom5", "mom20", "vol20", "ma_ratio", "rng"]
ADF_5PCT = -2.86          # ADF 检验 5% 临界值(常数项)


# ==================== 一、分数阶差分 ====================

def ffd_weights(d, thresh=1e-4, max_k=3000):
    """固定宽度分数阶差分的权重(López FFD)。w0=1, w_k=-w_{k-1}(d-k+1)/k。
    d∈(0,1) 时权重缓慢衰减=长记忆。返回'最老在前'的权重数组。"""
    w = [1.0]
    for k in range(1, max_k):
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
    return np.array(w[::-1])               # 反转成最老在前,便于 correlate


def frac_diff(series, d, thresh=1e-4):
    """对序列做分数阶差分(固定宽度窗)。d=0 原样, d=1 一阶差分, 中间=部分保留记忆。"""
    w = ffd_weights(d, thresh)
    width = len(w)
    vals = series.values.astype(float)
    out = np.full(len(vals), np.nan)
    if width <= len(vals):
        out[width - 1:] = np.correlate(vals, w, mode="valid")   # 每点=权重·历史窗
    return pd.Series(out, index=series.index)


# ==================== 二、三重栅栏事件(供 Purged CV) ====================

def make_features(close):
    r = np.log(close / close.shift(1))
    df = pd.DataFrame(index=close.index)
    df["mom5"] = r.rolling(5).sum()
    df["mom20"] = r.rolling(20).sum()
    df["vol20"] = r.rolling(20).std()
    df["ma_ratio"] = close / close.rolling(20).mean() - 1
    df["rng"] = r.rolling(10).max() - r.rolling(10).min()
    return df, r


def build_events(close, pt=1.5, sl=0.7, vert=VERT):
    """主信号(动量>0)处生成事件,三重栅栏贴标签;记录持有期[t0,t1]用于 purge。"""
    feat, r = make_features(close)
    sigma = r.rolling(20).std()
    logp = np.log(close.values)
    n = len(close)
    rows = []
    for i in range(60, n - vert):
        if not np.isfinite(feat["mom20"].iloc[i]) or feat["mom20"].iloc[i] <= 0:
            continue
        si = sigma.iloc[i]
        if not np.isfinite(si) or si <= 0:
            continue
        up, dn = pt * si * np.sqrt(vert), sl * si * np.sqrt(vert)
        p0 = logp[i]
        hit = vert
        for j in range(1, vert + 1):
            g = logp[i + j] - p0
            if g >= up or g <= -dn:
                hit = j
                break
        exret = logp[i + hit] - p0
        rows.append({"i": i, "t1": i + hit, "hit": hit,
                     **{c: feat[c].iloc[i] for c in FEAT_COLS},
                     "meta": int(exret > 0)})
    return pd.DataFrame(rows)


def cv_auc(ev, mode="shuffle", embargo=VERT, k=5):
    """三种交叉验证的样本外 AUC:
       shuffle = 乱序K折(初学者常犯,泄露最重) / block = 时间分块 / purge = 分块+清洗重叠+隔离。"""
    ev = ev.sort_values("i").reset_index(drop=True)
    X, y = ev[FEAT_COLS].values, ev["meta"].values
    t0, t1 = ev["i"].values, ev["t1"].values
    n = len(ev)
    if mode == "shuffle":
        folds = list(KFold(k, shuffle=True, random_state=0).split(X))
    else:
        idx = np.arange(n)
        folds = [(np.setdiff1d(idx, f), f) for f in np.array_split(idx, k)]
    aucs = []
    for train_idx, test_idx in folds:
        train_mask = np.zeros(n, bool); train_mask[train_idx] = True
        if mode == "purge":                       # 清洗:扔掉与测试期重叠的训练样本 + 隔离
            a, b = t0[test_idx].min(), t1[test_idx].max()
            overlap = ~((t1 < a) | (t0 > b + embargo))
            train_mask &= ~overlap
        if train_mask.sum() < 50 or len(np.unique(y[test_idx])) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20,
                                    random_state=0, n_jobs=1)
        rf.fit(X[train_mask], y[train_mask])
        p = rf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], p))
    return float(np.mean(aucs)) if aucs else np.nan


# ==================== 三、PBO via CSCV ====================

def pbo_cscv(M, S=14):
    """组合对称交叉验证求回测过拟合概率。M: (周期 × 策略) 的收益矩阵。
       把周期分 S 块,枚举一半作样本内(IS)一半作样本外(OOS):挑 IS 最优策略,看它 OOS 的相对排名。
       PBO = IS最优策略在 OOS 低于中位数的比例。返回 (PBO, logit λ 的分布)。"""
    T, N = M.shape
    blocks = np.array_split(np.arange(T), S)
    lams = []
    for is_blocks in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in is_blocks])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_blocks])
        is_perf = M[is_rows].sum(axis=0)
        oos_perf = M[oos_rows].sum(axis=0)
        n_star = int(np.argmax(is_perf))             # 样本内最优策略
        rank = int((oos_perf <= oos_perf[n_star]).sum())   # 它在 OOS 的排名(1..N)
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lams.append(np.log(omega / (1 - omega)))     # logit
    lams = np.array(lams)
    return float((lams < 0).mean()), lams


def main():
    print("=" * 64)
    print("  进阶·金融机器学习 ② 分数阶差分 + Purged CV + PBO（诚实三件套）")
    print("=" * 64)
    close = qt.load_sample_data()["Close"]
    logp = np.log(close)
    # 示例数据全程"先牛后熊"(驼峰形),整段反被 ADF 误判为平稳(也是个教训:ADF 会被驼峰骗)。
    # 取【趋势段】(前60%,单边上行)演示分数阶差分最清楚——真实长期上行的股价就是这样的非平稳。
    series = logp.iloc[:int(len(logp) * 0.6)]

    # ---- 实验① 分数阶差分: 平稳性 vs 记忆 ----
    ds = np.round(np.linspace(0, 1, 11), 2)
    adf_stats, corrs = [], []
    for d in ds:
        fd = frac_diff(series, d).dropna()
        stat = adfuller(fd, maxlag=1, regression="c", autolag=None)[0]
        corr = np.corrcoef(fd.values, series.loc[fd.index].values)[0, 1]
        adf_stats.append(stat); corrs.append(corr)
    adf_stats, corrs = np.array(adf_stats), np.array(corrs)
    passed = np.where(adf_stats < ADF_5PCT)[0]
    d_star = ds[passed[0]] if len(passed) else 1.0
    corr_star = corrs[passed[0]] if len(passed) else corrs[-1]
    print(f"① 分数阶差分: 让序列恰好平稳(ADF<5%临界)的最小 d* = {d_star}")
    print(f"   在 d*={d_star} 时与原始logP的相关性(记忆)仍有 {corr_star:.2f};")
    print(f"   而一阶差分 d=1(收益)和原价的相关性只有 {corrs[-1]:.2f}=记忆基本删光。\n")

    # ---- 实验③ CV 泄露 ----
    ev = build_events(close)
    # 标签重叠度:平均每个事件的持有期覆盖多少其它事件
    overlap_ratio = (ev["hit"].mean())
    auc_shuffle = cv_auc(ev, "shuffle")
    auc_block = cv_auc(ev, "block")
    auc_purge = cv_auc(ev, "purge")
    print(f"③ 三重栅栏共 {len(ev)} 个事件,平均持有 {overlap_ratio:.1f} 天→相邻标签严重重叠。")
    print(f"   交叉验证样本外 AUC:  乱序K折 {auc_shuffle:.3f}  →  分块 {auc_block:.3f}  →  分块+Purge+Embargo {auc_purge:.3f}")
    print(f"   乱序CV比诚实的Purged高 {auc_shuffle-auc_purge:+.3f}——这就是标签重叠'偷看未来'的虚高。\n")

    # ---- 实验④ PBO ----
    mkt = np.log(close / close.shift(1)).dropna().values
    T = len(mkt)
    n_periods, N = 50, 40
    # N 个"纯运气"策略:随机多空(0/1)持仓 × 市场收益
    sig = RNG.integers(0, 2, size=(T, N)).astype(float)
    strat_daily = sig * mkt[:, None]
    edges = np.linspace(0, T, n_periods + 1).astype(int)
    M_noise = np.array([strat_daily[edges[p]:edges[p + 1]].sum(axis=0) for p in range(n_periods)])
    pbo_noise, lams_noise = pbo_cscv(M_noise)
    # 对照:把其中 5 个换成"真有 edge"(每期稳定正漂移),看 PBO 是否下降
    M_edge = M_noise.copy()
    M_edge[:, :5] += 0.006 + 0.003 * np.abs(RNG.standard_normal((n_periods, 5)))
    pbo_edge, _ = pbo_cscv(M_edge)
    print(f"④ PBO 回测过拟合概率(CSCV, {N}个策略/{n_periods}周期):")
    print(f"   纯运气策略池: PBO = {pbo_noise:.0%}——挑'样本内最好'的,样本外≈掷硬币(随机自然≈50%)。")
    print(f"   掺入5个真有edge的策略后: PBO 降到 {pbo_edge:.0%}——PBO 能区分'真本事'和'过拟合'。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    # 图① 平稳性 vs 记忆
    ax = axes[0, 0]
    ax.plot(ds, adf_stats, "o-", color="#1f77b4", label="ADF stat (lower = stationary)")
    ax.axhline(ADF_5PCT, color="#d62728", ls="--", lw=1.2, label="ADF 5% critical")
    ax.axvline(d_star, color="#2ca02c", lw=1.2, alpha=0.8)
    ax.set_xlabel("differencing order d"); ax.set_ylabel("ADF statistic", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(ds, corrs, "s-", color="#ff7f0e", label="corr with original (memory)")
    ax2.set_ylabel("correlation with log-price", color="#ff7f0e")
    ax2.scatter([d_star], [corr_star], s=110, color="#2ca02c", zorder=5,
                label=f"d*={d_star}: stationary & memory {corr_star:.2f}")
    ax.set_title("(1) Fractional diff: stationarity vs memory trade-off")
    ax.grid(alpha=0.3)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=7.5, loc="center right")

    # 图② 三条序列(同样用趋势段)
    ax = axes[0, 1]
    def z(s): s = s.dropna(); return (s - s.mean()) / s.std()
    zp = z(series); zf = z(frac_diff(series, d_star)); zd = z(series.diff())
    ax.plot(zp.index, zp.values, color="#1f77b4", lw=1.0, label="raw log-price (non-stationary)")
    ax.plot(zf.index, zf.values, color="#2ca02c", lw=1.0, label=f"frac-diff d*={d_star} (stationary + memory)")
    ax.plot(zd.index, zd.values, color="#bbbbbb", lw=0.7, alpha=0.7, label="full diff d=1 (stationary, no memory)")
    ax.set_title("(2) Frac-diff keeps the shape; full diff is just noise")
    ax.set_ylabel("z-score"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # 图③ CV 泄露
    ax = axes[1, 0]
    names = ["shuffled\nK-fold\n(leaky)", "blocked\nCV", "blocked +\nPurge +\nEmbargo\n(honest)"]
    vals = [auc_shuffle, auc_block, auc_purge]
    cols = ["#d62728", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(names, vals, color=cols)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="AUC 0.5 = no skill")
    ax.set_ylim(0.45, max(vals) + 0.05)
    ax.set_title("(3) Overlapping labels leak: naive CV is inflated")
    ax.set_ylabel("out-of-sample AUC"); ax.legend(fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 图④ PBO
    ax = axes[1, 1]
    ax.hist(lams_noise, bins=40, density=True, color="#9467bd", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", lw=1.5)
    ax.fill_betweenx([0, ax.get_ylim()[1]], lams_noise.min(), 0, color="#d62728", alpha=0.12)
    ax.text(0.03, 0.92, f"PBO (noise pool) = {pbo_noise:.0%}\nwith real-edge strats: {pbo_edge:.0%}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800"))
    ax.set_title("(4) PBO: in-sample-best is a coin flip out-of-sample")
    ax.set_xlabel("logit λ  (λ<0 = IS-best below OOS median)"); ax.set_ylabel("density")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "诚实三件套配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（分数差分/序列对比/CV泄露/PBO 四合一）")
    print("=" * 64)


if __name__ == "__main__":
    main()
