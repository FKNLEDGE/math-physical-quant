"""
进阶·金融机器学习 ③ 特征重要性的陷阱（MDI / MDA / SFI）
==========================================================
[D3](路线D)说过树模型能给"特征重要性"。López de Prado 警告:这些数字会骗人,
尤其在金融这种低信噪比、特征高度相关的场景。这一课用一个【已知真相】的受控实验拆穿它。

  我们故意造三类特征(用 make_classification, 列顺序已知):
    · informative(0,1,2): 真正驱动 y 的
    · redundant(3,4):    informative 的线性组合(高度相关="替身")
    · noise(5-9):        纯随机,和 y 无关

  三种重要性方法各有各的坑:
    MDI 均值不纯度下降(树自带, 样本内): 给【噪声】也打分(偏差),相关特征互相稀释
    MDA 置换重要性(打乱某列看样本外掉多少): 噪声→0(好),但相关"替身"互相遮蔽→都显得不重要
    SFI 单特征重要性(每个特征单独训练): 避开替身效应、还原冗余特征真实价值,但看不到交互

  实验①  MDI: 噪声也被打了分(假阳性陷阱)
  实验②  MDA: 噪声归零,但相关替身互相遮蔽(都被低估)
  实验③  SFI: 冗余特征的真实价值被还原(无替身效应)
  实验④  三法对三类特征的平均打分对比——没有一种方法单独可信

运行：python feature_importance_traps.py
依赖：numpy, scikit-learn, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RNG = 0
N_INF, N_RED, N_NOISE = 3, 2, 5
LABELS = ([f"inf{i}" for i in range(N_INF)] +
          [f"red{i}" for i in range(N_RED)] +
          [f"noise{i}" for i in range(N_NOISE)])
TYPES = ["informative"] * N_INF + ["redundant"] * N_RED + ["noise"] * N_NOISE
COLORS = {"informative": "#2ca02c", "redundant": "#ff7f0e", "noise": "#d62728"}


def main():
    print("=" * 64)
    print("  进阶·金融机器学习 ③ 特征重要性的陷阱（MDI / MDA / SFI）")
    print("=" * 64)
    # shuffle=False → 列顺序固定: [informative | redundant | noise]
    X, y = make_classification(n_samples=4000, n_features=N_INF + N_RED + N_NOISE,
                               n_informative=N_INF, n_redundant=N_RED, n_repeated=0,
                               n_clusters_per_class=1, class_sep=1.0, shuffle=False,
                               random_state=RNG)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=RNG)
    # 故意用【深/不正则】的森林:这正是 MDI 陷阱最严重的场景(过拟合→给噪声乱发分)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1,
                                random_state=RNG, n_jobs=1).fit(Xtr, ytr)
    print(f"  特征: {N_INF}个informative + {N_RED}个redundant(相关替身) + {N_NOISE}个noise;"
          f" 模型样本外 AUC {roc_auc_score(yte, rf.predict_proba(Xte)[:,1]):.3f}\n")

    # --- MDI: 树自带的不纯度重要性(样本内) ---
    mdi = rf.feature_importances_

    # --- MDA: 置换重要性(样本外) ---
    mda = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=RNG, n_jobs=1).importances_mean
    mda = np.clip(mda, 0, None)

    # --- SFI: 每个特征单独训练,看它一个人的样本外 AUC(减去0.5无技能基线) ---
    sfi = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        m = RandomForestClassifier(n_estimators=120, max_depth=4, min_samples_leaf=20,
                                   random_state=RNG, n_jobs=1).fit(Xtr[:, [j]], ytr)
        sfi[j] = roc_auc_score(yte, m.predict_proba(Xte[:, [j]])[:, 1]) - 0.5

    def avg_by_type(imp):
        return {t: imp[np.array(TYPES) == t].mean() for t in ["informative", "redundant", "noise"]}

    for name, imp in [("MDI(样本内不纯度)", mdi), ("MDA(置换,样本外)", mda), ("SFI(单特征,样本外)", sfi)]:
        a = avg_by_type(imp)
        print(f"  {name:22s} 平均分: informative {a['informative']:.3f} | "
              f"redundant {a['redundant']:.3f} | noise {a['noise']:.3f}")
    print()
    print(f"① MDI 把 {avg_by_type(mdi)['noise']/mdi.sum()*100*N_NOISE:.0f}% 总重要性发给了纯噪声(假阳性陷阱)。")
    print(f"② MDA 噪声≈{avg_by_type(mda)['noise']:.3f}(几乎归零,好);但相关替身被互相遮蔽→低估。")
    print(f"③ SFI 还原冗余特征的真实价值: redundant {avg_by_type(sfi)['redundant']:.3f} ≈ informative "
          f"{avg_by_type(sfi)['informative']:.3f}(单独看它们都很能打)。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    xpos = np.arange(len(LABELS))
    bar_colors = [COLORS[t] for t in TYPES]

    def bar_panel(ax, imp, title, ylab):
        ax.bar(xpos, imp, color=bar_colors)
        ax.set_xticks(xpos); ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y")

    bar_panel(axes[0, 0], mdi, "(1) MDI (in-sample impurity): noise gets scored too!", "importance")
    axes[0, 0].axhline(mdi[np.array(TYPES) == "noise"].max(), color="k", ls=":", lw=1,
                       label="max noise score (false positive)")
    axes[0, 0].legend(fontsize=7.5)

    bar_panel(axes[0, 1], mda, "(2) MDA (permutation, OOS): noise~0, but twins mask each other", "accuracy drop")
    axes[1, 0].axhline(0, color="k", lw=0.6)
    bar_panel(axes[1, 0], sfi, "(3) SFI (single-feature, OOS): redundant value restored", "AUC − 0.5")

    # 图④ 三法 × 三类 的平均分对比
    ax = axes[1, 1]
    methods = ["MDI", "MDA", "SFI"]
    imps = [mdi, mda, sfi]
    types = ["informative", "redundant", "noise"]
    # 每种方法内部归一化(除以informative均值)便于跨方法比较形状
    width = 0.25
    mx = np.arange(len(methods))
    for k, t in enumerate(types):
        vals = []
        for imp in imps:
            a = avg_by_type(imp)
            base = a["informative"] if a["informative"] > 1e-9 else 1.0
            vals.append(a[t] / base)
        ax.bar(mx + (k - 1) * width, vals, width, color=COLORS[t], label=t)
    ax.set_xticks(mx); ax.set_xticklabels(methods)
    ax.set_title("(4) No single method is trustworthy (normalized to informative=1)")
    ax.set_ylabel("avg importance / informative"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.annotate("MDI: noise > 0\n(false positive)", xy=(0 + width, vals[-1] if False else 0.0),
                xytext=(-0.35, 0.6), fontsize=7.5, color="#d62728")
    ax.annotate("MDA: redundant\nunderrated", xy=(1, 0), xytext=(0.7, 0.45), fontsize=7.5, color="#ff7f0e")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "特征重要性配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（MDI/MDA/SFI/三法对比 四合一）")
    print("=" * 64)


if __name__ == "__main__":
    main()
