"""
D3 · 树模型与特征重要性（量化最实用的一类模型）
================================================
决策树=不断问『是/否』把特征空间切成小块；它的集成(随机森林、梯度提升)
是表格/量化数据的主力武器。但在金融的低信噪比下，仍要万分警惕。

  实验①  单棵决策树：把平面切成矩形区域(可解释,但易过拟合)
  实验②  随机森林：很多棵树投票 → 边界更平滑(bagging 降方差)
  实验③  bagging 的威力：树越多，样本外越稳(降方差到地板)
  实验④  特征重要性的陷阱：连纯噪声特征，森林也会给出『重要性』排名

运行：python tree_models.py
依赖：numpy, scikit-learn, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(0)


def plot_boundary(ax, model, X, y, title):
    x0 = np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200)
    x1 = np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200)
    xx, yy = np.meshgrid(x0, x1)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap="coolwarm", edgecolor="k", linewidth=0.2)
    ax.set_title(title); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")


def main():
    print("=" * 60)
    print("  D3 · 树模型与特征重要性")
    print("=" * 60)
    print("  决策树：递归地问『某特征 > 某阈值?』，把数据切成越来越纯的小块。")
    print("  优点:处理非线性/交互、无需缩放、可解释；缺点:单棵深树严重过拟合。\n")

    X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 单棵(深)决策树：过拟合的破碎边界
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    plot_boundary(axes[0, 0], tree, Xtr, ytr,
                  f"(1) Single full tree (overfit), train acc={tree.score(Xtr,ytr):.2f}")
    print(f"① 单棵全树：训练准确率 {tree.score(Xtr,ytr):.2f}(完美), 测试 {tree.score(Xte,yte):.2f}")
    print("   边界破碎、为每个噪声点'画'出小岛——高方差过拟合(闭环D1)。\n")

    # 图② 随机森林：投票平滑
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    plot_boundary(axes[0, 1], rf, Xtr, ytr,
                  f"(2) Random forest (200 trees), test acc={rf.score(Xte,yte):.2f}")
    print(f"② 随机森林(200棵)：测试准确率 {rf.score(Xte,yte):.2f}，边界平滑得多。")
    print("   每棵树用自助样本+随机特征子集 → 投票平均，把方差降下来(bagging)。\n")

    # 图③ bagging：树越多越稳
    ax = axes[1, 0]
    ns = [1, 2, 5, 10, 25, 50, 100, 200, 400]
    accs = [RandomForestClassifier(n_estimators=k, random_state=0).fit(Xtr, ytr).score(Xte, yte) for k in ns]
    ax.plot(ns, accs, "o-", color="#2ca02c")
    ax.set_xscale("log")
    ax.set_title("(3) Bagging: more trees -> more stable test accuracy")
    ax.set_xlabel("number of trees (log)"); ax.set_ylabel("test accuracy")
    ax.grid(alpha=0.3)
    print(f"③ bagging：1棵树测试{accs[0]:.2f} → 400棵{accs[-1]:.2f}，更稳。降方差到地板后趋平。\n")

    # 图④ 特征重要性陷阱：纯噪声也能排出'重要性'
    ax = axes[1, 1]
    n, p = 300, 8
    Xn = RNG.standard_normal((n, p))
    y_noise = RNG.integers(0, 2, n)                 # 目标=纯噪声,和特征无关
    rf_noise = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xn, y_noise)
    imp = rf_noise.feature_importances_
    ax.bar(range(p), imp, color="#d62728")
    ax.set_title("(4) Feature importance on PURE NOISE (all spurious)")
    ax.set_xlabel("feature #"); ax.set_ylabel("importance")
    ax.set_xticks(range(p)); ax.grid(alpha=0.3, axis="y")
    print("④ 特征重要性陷阱：目标设成【纯噪声】(与特征无关)，森林照样给出'重要性'排名，")
    print(f"   最高的特征#{int(np.argmax(imp))}重要性{imp.max():.3f} vs 最低{imp.min():.3f}——全是假的。")
    print("   在金融低信噪比下,'重要特征'极可能就是这样的噪声。重要性≠因果,要交叉验证。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "D3配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（单树/森林/bagging/重要性陷阱 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
