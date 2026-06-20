"""
D1 · 监督学习与偏差-方差权衡（机器学习第一性原理）
=================================================
机器学习的目标不是『背下训练数据』，而是『在没见过的新数据上也准』(泛化)。
所有 ML 的核心张力，就是偏差-方差权衡：模型太简单(欠拟合)还是太复杂(过拟合)。

  实验①  用不同复杂度的多项式拟合：欠拟合 / 恰好 / 过拟合
  实验②  偏差-方差曲线：训练误差一路降，测试误差先降后升(U形)
  实验③  高方差=不稳定：复杂模型在不同样本上拟合结果天差地别
  实验④  正则化：给复杂度加惩罚，把过拟合拉回来

运行：python bias_variance.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(0)


def true_f(x):
    """真实规律(我们想学的)：一条光滑曲线。"""
    return np.sin(1.6*np.pi*x)


def make_data(n, noise=0.35, seed=None):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = true_f(x) + rng.normal(0, noise, n)
    return x, y


def poly_fit_predict(x, y, degree, x_eval, lam=0.0):
    """多项式拟合(可带岭正则 lam)，返回在 x_eval 上的预测。"""
    X = np.vander(x, degree+1)
    Xe = np.vander(x_eval, degree+1)
    A = X.T @ X + lam*np.eye(degree+1)
    w = np.linalg.solve(A, X.T @ y)
    return Xe @ w


def mse(a, b):
    return np.mean((a-b)**2)


def main():
    print("=" * 60)
    print("  D1 · 监督学习与偏差-方差权衡")
    print("=" * 60)
    print("  监督学习 = 从样本里学一个函数 f: 特征X → 目标y，并能预测【新】数据。")
    print("  量化里：X=各种指标/因子，y=未来收益(回归) 或 涨跌(分类)。")
    print("  核心：目标是【泛化】(没见过的数据上准)，不是【记忆】(背下训练集)。\n")

    x_tr, y_tr = make_data(20, seed=1)
    x_grid = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 不同复杂度的拟合
    ax = axes[0, 0]
    ax.scatter(x_tr, y_tr, color="k", s=25, zorder=5, label="training data (noisy)")
    ax.plot(x_grid, true_f(x_grid), "g-", lw=2, label="true pattern")
    for deg, c in [(1, "#1f77b4"), (4, "#2ca02c"), (15, "#d62728")]:
        ax.plot(x_grid, poly_fit_predict(x_tr, y_tr, deg, x_grid), "--", color=c,
                label=f"degree {deg}")
    ax.set_ylim(-2, 2)
    ax.set_title("(1) Underfit (deg1) / good (deg4) / overfit (deg15)")
    ax.set_xlabel("feature x"); ax.set_ylabel("target y")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    print("① 1次=欠拟合(太简单,抓不住规律=高偏差)；4次=恰好；15次=过拟合(穿过每个点,学了噪声=高方差)\n")

    # 图② 偏差-方差曲线
    ax = axes[0, 1]
    x_te, y_te = make_data(500, seed=99)         # 大测试集近似真值
    degs = np.arange(1, 16)
    tr_err, te_err = [], []
    for d in degs:
        tr_err.append(mse(poly_fit_predict(x_tr, y_tr, d, x_tr), y_tr))
        te_err.append(mse(poly_fit_predict(x_tr, y_tr, d, x_te), y_te))
    ax.plot(degs, tr_err, "o-", color="#1f77b4", label="training error (keeps dropping)")
    ax.plot(degs, te_err, "s-", color="#d62728", label="test error (U-shape!)")
    best = degs[np.argmin(te_err)]
    ax.axvline(best, color="gray", ls="--", label=f"sweet spot deg={best}")
    ax.set_yscale("log")
    ax.set_title("(2) Bias-variance: test error is U-shaped")
    ax.set_xlabel("model complexity (degree)"); ax.set_ylabel("MSE (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"② 训练误差随复杂度【一路降】(背得越来越熟)，但测试误差【先降后升】(U形)。")
    print(f"   最优复杂度在中间(deg={best})。左边欠拟合(高偏差)，右边过拟合(高方差)。\n")

    # 图③ 高方差 = 不稳定
    ax = axes[1, 0]
    ax.plot(x_grid, true_f(x_grid), "g-", lw=2, label="true")
    for i in range(8):
        xi, yi = make_data(20, seed=100+i)        # 8 个不同的训练样本
        ax.plot(x_grid, poly_fit_predict(xi, yi, 15, x_grid), color="#d62728", alpha=0.4, lw=0.8)
    ax.plot([], [], color="#d62728", label="degree 15 on 8 samples")
    ax.set_ylim(-2.5, 2.5)
    ax.set_title("(3) High variance: complex model = wildly unstable fits")
    ax.set_xlabel("feature x"); ax.set_ylabel("y"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("③ 高方差=不稳定：同一个15次模型，换个训练样本，拟合曲线天差地别。")
    print("   稳不住 → 换段数据就失效。这正是过拟合在『信号vs噪声』上的表现。\n")

    # 图④ 正则化驯服过拟合
    ax = axes[1, 1]
    ax.scatter(x_tr, y_tr, color="k", s=25, zorder=5, label="data")
    ax.plot(x_grid, true_f(x_grid), "g-", lw=2, label="true")
    ax.plot(x_grid, poly_fit_predict(x_tr, y_tr, 15, x_grid, lam=0), "--", color="#d62728", label="deg15, no penalty")
    ax.plot(x_grid, poly_fit_predict(x_tr, y_tr, 15, x_grid, lam=1e-2), "-", color="#1f77b4", lw=2, label="deg15 + ridge penalty")
    ax.set_ylim(-2, 2)
    ax.set_title("(4) Regularization tames overfitting")
    ax.set_xlabel("feature x"); ax.set_ylabel("y"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("④ 正则化：给『复杂度/大系数』加惩罚(岭回归 λ)，强迫模型更平滑。")
    print("   同样15次多项式，加了惩罚后从疯狂扭动变回光滑——可控的复杂度。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "D1配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（拟合/偏差方差曲线/高方差/正则化 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
