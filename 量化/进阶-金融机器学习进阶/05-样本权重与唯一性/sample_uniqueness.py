"""
进阶·金融机器学习 ⑤ 样本权重与标签唯一性
============================================
[① 三重栅栏](../01-三重栅栏与元标注/) 的标签由未来一段路径决定→相邻样本的标签【在时间上重叠】。
[② Purged CV](../02-分数阶差分与PurgedCV/) 处理了"重叠导致验证泄露";这一课处理另一面:

   标签重叠 → 样本【彼此不独立】→ 标准 ML 把它们当独立样本,等于【重复计数】、过拟合。

López 的解药:按"唯一性"给样本【加权/重采样】——重叠多的样本被【降权】。

  关键量:
    并发度 c_t = 时刻 t 有几个标签在"活动"(重叠)
    唯一性 u_i = 样本 i 的标签期内 1/c_t 的平均(重叠越多越小)
    有效样本量 = Σ u_i  ——你以为有 N 个样本,其实只有这么多"独立"的!

  实验①  标签并发度随时间:同一时刻常有好几个标签在活动(重叠)
  实验②  唯一性分布 + 有效样本量 ≪ 原始 N(过拟合的隐形来源)
  实验③  序贯自助 vs 普通自助:序贯采样挑出更"唯一"的样本(少冗余)
  实验④  用唯一性加权训练 vs 不加权:样本外表现(诚实对比)

运行：python sample_uniqueness.py
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
from sklearn.metrics import roc_auc_score
import quant_tools as qt

RNG = np.random.default_rng(0)
PT, SL, VERT = 1.5, 0.7, 20
FEATS = ["mom5", "mom20", "vol20", "ma_ratio", "rng"]


def build_events(close):
    """动量主信号处建三重栅栏事件,记录持有期 [i, t1] 与特征/标签。"""
    r = np.log(close / close.shift(1))
    f = pd.DataFrame(index=close.index)
    f["mom5"] = r.rolling(5).sum(); f["mom20"] = r.rolling(20).sum()
    f["vol20"] = r.rolling(20).std(); f["ma_ratio"] = close / close.rolling(20).mean() - 1
    f["rng"] = r.rolling(10).max() - r.rolling(10).min()
    sigma = r.rolling(20).std(); logp = np.log(close.values); n = len(close)
    rows = []
    for i in range(60, n - VERT):
        if not np.isfinite(f["mom20"].iloc[i]) or f["mom20"].iloc[i] <= 0:
            continue
        si = sigma.iloc[i]
        if not np.isfinite(si) or si <= 0:
            continue
        up, dn = PT * si * np.sqrt(VERT), SL * si * np.sqrt(VERT)
        p0 = logp[i]; hit = VERT
        for j in range(1, VERT + 1):
            g = logp[i + j] - p0
            if g >= up or g <= -dn:
                hit = j; break
        rows.append({"i": i, "t1": i + hit, **{c: f[c].iloc[i] for c in FEATS},
                     "y": int(logp[i + hit] - p0 > 0)})
    return pd.DataFrame(rows), n


def concurrency_uniqueness(t0, t1, nbars):
    """计算每根 bar 的并发度,和每个样本的平均唯一性。"""
    conc = np.zeros(nbars)
    for a, b in zip(t0, t1):
        conc[a:b + 1] += 1
    u = np.array([np.mean(1.0 / conc[a:b + 1]) for a, b in zip(t0, t1)])
    return conc, u


def avg_uniqueness_of_set(sel, I):
    """给定被选样本集合,算它们的平均唯一性(用指示矩阵 I[bars,events])。"""
    c = I[:, sel].sum(axis=1)
    return np.mean([np.mean(1.0 / c[I[:, k] > 0]) for k in sel])


def sequential_bootstrap(I, draws, rng):
    """序贯自助:每次按'给定已选后的唯一性'概率抽样,主动避开冗余(重叠)样本。"""
    nbars, m = I.shape
    concd = np.zeros(nbars); picked = []
    for _ in range(draws):
        au = np.array([np.mean(1.0 / (concd[I[:, k] > 0] + 1)) if I[:, k].any() else 0.0
                       for k in range(m)])
        p = au / au.sum()
        j = rng.choice(m, p=p); picked.append(j); concd[I[:, j] > 0] += 1
    return picked


def main():
    print("=" * 60)
    print("  进阶·金融机器学习 ⑤ 样本权重与标签唯一性")
    print("=" * 60)
    close = qt.load_sample_data()["Close"]
    ev, nbars = build_events(close)
    t0, t1 = ev["i"].values, ev["t1"].values
    N = len(ev)
    conc, u = concurrency_uniqueness(t0, t1, nbars)
    eff = u.sum()
    print(f"  三重栅栏事件 {N} 个;活动时段平均并发度 {conc[conc>0].mean():.1f}")
    print(f"① 标签在时间上大量重叠(并发度>1)——样本彼此不独立")
    print(f"② 平均唯一性 {u.mean():.2f};有效样本量 Σu = {eff:.0f}（你以为有 {N} 个,其实只有 ~{eff:.0f} 个独立!）\n")

    # ③ 序贯 vs 普通自助(取子集控时长)
    m = min(250, N)
    sub = RNG.choice(N, m, replace=False)
    I = np.zeros((nbars, m))
    for k, idx in enumerate(sub):
        I[t0[idx]:t1[idx] + 1, k] = 1
    seq = sequential_bootstrap(I, m, RNG)
    std = RNG.choice(m, m, replace=True)
    au_seq, au_std = avg_uniqueness_of_set(seq, I), avg_uniqueness_of_set(std, I)
    print(f"③ 自助采样平均唯一性: 序贯 {au_seq:.2f} > 普通 {au_std:.2f}（序贯主动避开冗余,样本更独立）\n")

    # ④ 唯一性加权训练 vs 不加权
    cut = int(N * 0.6)
    tr, te = ev.iloc[:cut], ev.iloc[cut:]
    Xtr, ytr = tr[FEATS].values, tr["y"].values
    Xte, yte = te[FEATS].values, te["y"].values
    wtr = u[:cut]
    aucs_u, aucs_p = [], []
    for s in range(10):
        m1 = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=20,
                                    random_state=s, n_jobs=1).fit(Xtr, ytr)
        m2 = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=20,
                                    random_state=s, n_jobs=1).fit(Xtr, ytr, sample_weight=wtr)
        aucs_p.append(roc_auc_score(yte, m1.predict_proba(Xte)[:, 1]))
        aucs_u.append(roc_auc_score(yte, m2.predict_proba(Xte)[:, 1]))
    print(f"④ 样本外 AUC(10 次平均): 不加权 {np.mean(aucs_p):.3f} | 唯一性加权 {np.mean(aucs_u):.3f}")
    print("   低信噪比下提升常常微小,但加权是'诚实对待非独立样本'的正确做法(别把冗余当新证据)。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.fill_between(np.arange(nbars), conc, color="#1f77b4", alpha=0.7, lw=0)
    ax.axhline(conc[conc > 0].mean(), color="#d62728", ls="--", lw=1.2,
               label=f"avg concurrency {conc[conc>0].mean():.1f}")
    ax.set_title("(1) Label concurrency: many labels overlap in time")
    ax.set_xlabel("time (bar)"); ax.set_ylabel("# active labels"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(u, bins=30, color="#9467bd", edgecolor="white")
    ax.axvline(u.mean(), color="#d62728", ls="--", lw=1.5, label=f"mean uniqueness {u.mean():.2f}")
    ax.set_title(f"(2) Uniqueness: effective N = Σu ≈ {eff:.0f} (not {N}!)")
    ax.set_xlabel("average uniqueness u"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.text(0.5, 0.78, f"you THINK {N} samples\nyou HAVE ~{eff:.0f} independent",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800"))

    ax = axes[1, 0]
    ax.bar(["standard\nbootstrap", "sequential\nbootstrap"], [au_std, au_seq],
           color=["#888", "#2ca02c"])
    ax.set_title("(3) Sequential bootstrap draws more-unique (less redundant) samples")
    ax.set_ylabel("avg uniqueness of drawn set"); ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate([au_std, au_seq]):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=10)

    ax = axes[1, 1]
    ax.bar(["unweighted", "uniqueness-\nweighted"], [np.mean(aucs_p), np.mean(aucs_u)],
           yerr=[np.std(aucs_p), np.std(aucs_u)], color=["#888", "#1f77b4"], capsize=4)
    ax.axhline(0.5, color="k", ls=":", lw=1, label="AUC 0.5 = no skill")
    ax.set_ylim(0.40, max(np.mean(aucs_p), np.mean(aucs_u)) + 0.06)
    ax.set_title("(4) Uniqueness-weighted training (out-of-sample AUC)")
    ax.set_ylabel("OOS AUC"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "样本唯一性配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（并发度/唯一性/自助/加权 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
