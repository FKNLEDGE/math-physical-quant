"""
进阶·金融机器学习 ④ 聚类特征重要性（解决"替身遮蔽"）
========================================================
[③ 特征重要性陷阱](../03-特征重要性陷阱/) 留了个坑:MDA(置换重要性)会被【替身效应】坑——
两个高度相关的特征,打乱其一时模型用另一个顶上,于是【双双显得不重要】,真信号被埋没。

López de Prado 的解药很直接:**先把相关特征【聚类】,再【整组一起打乱】算重要性。**
相关的"替身"被分进同一组,组内不再互相顶替→它们的真实(联合)重要性就显出来了。

  我们用和 ③ 同一份已知真相的数据(3 informative + 2 redundant + 5 noise)。

  实验①  按相关性层次聚类:informative 和它的 redundant 替身被分进【同一组】
  实验②  单特征 MDA(③ 的做法):redundant 被遮蔽,显得不重要(低)
  实验③  聚类 MDA(整组打乱):含信号的那组拿到【完整的高重要性】,噪声组≈0
  实验④  对比:信号组"单特征之和(被遮蔽)" vs "整组联合(还原)"——聚类把真信号救回来

运行：python clustered_feature_importance.py
依赖：numpy, scipy, scikit-learn, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
NI, NR, NN = 3, 2, 5
LABELS = [f"inf{i}" for i in range(NI)] + [f"red{i}" for i in range(NR)] + [f"noise{i}" for i in range(NN)]
TYPES = ["informative"] * NI + ["redundant"] * NR + ["noise"] * NN
COLORS = {"informative": "#2ca02c", "redundant": "#ff7f0e", "noise": "#d62728"}


def main():
    print("=" * 62)
    print("  进阶·金融机器学习 ④ 聚类特征重要性（解决替身遮蔽）")
    print("=" * 62)
    X, y = make_classification(n_samples=4000, n_features=NI + NR + NN,
                               n_informative=NI, n_redundant=NR, n_repeated=0,
                               n_clusters_per_class=1, class_sep=1.0, shuffle=False, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1,
                                random_state=0, n_jobs=1).fit(Xtr, ytr)
    base = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])

    # ① 按相关性层次聚类
    corr = np.corrcoef(X.T)
    dist = 1 - np.abs(corr)
    Z = linkage(squareform(dist, checks=False), method="average")
    clusters = fcluster(Z, t=0.5, criterion="distance")
    print("① 聚类结果(相关→同组):")
    for c in sorted(set(clusters)):
        cols = [j for j in range(len(LABELS)) if clusters[j] == c]
        print(f"   组{c}: {[LABELS[j] for j in cols]}")
    print()

    # ② 单特征 MDA(③)
    def mda_single(j, reps=20):
        d = []
        for _ in range(reps):
            Xp = Xte.copy(); Xp[:, j] = RNG.permutation(Xp[:, j])
            d.append(base - roc_auc_score(yte, rf.predict_proba(Xp)[:, 1]))
        return max(np.mean(d), 0)
    single = np.array([mda_single(j) for j in range(len(LABELS))])

    # ③ 聚类 MDA(整组联合打乱)
    def mda_cluster(cols, reps=20):
        d = []
        for _ in range(reps):
            Xp = Xte.copy(); perm = RNG.permutation(len(Xte))
            for c in cols:
                Xp[:, c] = Xte[perm, c]           # 整组用【同一个】置换→一起打乱
            d.append(base - roc_auc_score(yte, rf.predict_proba(Xp)[:, 1]))
        return max(np.mean(d), 0)
    cl_ids = sorted(set(clusters))
    cl_cols = {c: [j for j in range(len(LABELS)) if clusters[j] == c] for c in cl_ids}
    cl_imp = {c: mda_cluster(cl_cols[c]) for c in cl_ids}

    # 找"信号组"(含 informative/redundant 的那组)
    sig_c = max(cl_ids, key=lambda c: sum(TYPES[j] != "noise" for j in cl_cols[c]))
    sig_single_sum = sum(single[j] for j in cl_cols[sig_c])
    print(f"② 单特征 MDA: redundant 被遮蔽,平均 {single[NI:NI+NR].mean():.3f}(远低于 informative {single[:NI].mean():.3f})")
    print(f"③ 聚类 MDA: 含信号的组 {[LABELS[j] for j in cl_cols[sig_c]]} → 重要性 {cl_imp[sig_c]:.3f}")
    print(f"④ 还原: 该组'单特征之和(遮蔽)' {sig_single_sum:.3f} → '整组联合' {cl_imp[sig_c]:.3f}——聚类救回真信号。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    # 图① 树状图
    ax = axes[0, 0]
    dendrogram(Z, labels=LABELS, ax=ax, color_threshold=0.5,
               leaf_font_size=8, leaf_rotation=45)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_title("(1) Cluster features by correlation: redundant joins its informative")
    ax.set_ylabel("distance = 1 − |corr|")

    # 图② 单特征 MDA
    ax = axes[0, 1]
    ax.bar(range(len(LABELS)), single, color=[COLORS[t] for t in TYPES])
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7.5)
    ax.set_title("(2) Single-feature MDA: redundant masked (substitution)")
    ax.set_ylabel("accuracy drop"); ax.grid(alpha=0.3, axis="y")

    # 图③ 聚类 MDA
    ax = axes[1, 0]
    names = [f"C{c}\n" + "+".join(LABELS[j][:4] for j in cl_cols[c][:3]) + ("…" if len(cl_cols[c]) > 3 else "")
             for c in cl_ids]
    vals = [cl_imp[c] for c in cl_ids]
    # 组颜色:含信号→绿,纯噪声→红
    cc = ["#2ca02c" if any(TYPES[j] != "noise" for j in cl_cols[c]) else "#d62728" for c in cl_ids]
    ax.bar(range(len(cl_ids)), vals, color=cc)
    ax.set_xticks(range(len(cl_ids))); ax.set_xticklabels(names, fontsize=7)
    ax.set_title("(3) Clustered MDA: the signal cluster gets full importance")
    ax.set_ylabel("accuracy drop (cluster)"); ax.grid(alpha=0.3, axis="y")

    # 图④ 还原对比
    ax = axes[1, 1]
    ax.bar(["single-feature\nsum (masked)", "clustered\n(joint, restored)"],
           [sig_single_sum, cl_imp[sig_c]], color=["#888", "#2ca02c"])
    ax.set_title("(4) Clustering rescues the masked signal")
    ax.set_ylabel("importance of signal cluster"); ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate([sig_single_sum, cl_imp[sig_c]]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "聚类重要性配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（树状图/单特征MDA/聚类MDA/还原 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
