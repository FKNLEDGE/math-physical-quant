"""
D2 · 金融为何是机器学习地狱（三大杀手 + 交叉验证陷阱）
=====================================================
机器学习在『信号强、数据多』的地方(图像/语言)所向披靡；金融恰是反面极端。

  杀手①  低信噪比：纯噪声里，灵活模型也能『拟合』出高样本内R²，样本外=0
  杀手②  信号埋在噪声里：金融的『可解释方差』低得吓人(R²≈0.01 vs 视觉0.95)
  杀手③  非平稳：训练时的规律，测试时可能反转(机制转换)→ 模型崩
  陷阱   交叉验证泄漏：时间序列上『打乱k折』会用未来预测过去，给出虚高分数

运行：python finance_ml_hell.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(0)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res/ss_tot


def ols_fit_pred(Xtr, ytr, Xte):
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xtr @ w, Xte @ w


def knn_time(t_tr, y_tr, t_te, k=3):
    """对一维特征 t 做 KNN：预测 = k 个最近邻的平均。"""
    preds = []
    for t in t_te:
        idx = np.argsort(np.abs(t_tr - t))[:k]
        preds.append(y_tr[idx].mean())
    return np.array(preds)


def main():
    print("=" * 60)
    print("  D2 · 金融为何是机器学习地狱")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 杀手① 纯噪声也能被『拟合』
    ax = axes[0, 0]
    n = 60
    y = RNG.standard_normal(2*n)                 # 目标=纯噪声(没有任何信号)
    ks = range(1, 56)
    r2_in, r2_out = [], []
    for kf in ks:
        X = RNG.standard_normal((2*n, kf))       # kf 个随机无用特征
        yin, yout = ols_fit_pred(X[:n], y[:n], X[n:])
        r2_in.append(r2(y[:n], yin))
        r2_out.append(max(r2(y[n:], yout), -1))
    ax.plot(list(ks), r2_in, color="#d62728", label="in-sample R^2")
    ax.plot(list(ks), r2_out, color="#1f77b4", label="out-of-sample R^2")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(1) Fitting PURE NOISE: in-sample R^2 -> 1, OOS ~ 0")
    ax.set_xlabel("number of (useless) features"); ax.set_ylabel("R^2")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("① 低信噪比：目标设成【纯噪声】(无任何信号)，喂越多随机特征，")
    print(f"   样本内R²一路升到 {r2_in[-1]:.2f}(把噪声也'拟合'进去)，样本外R²始终≈0甚至为负。")
    print("   → 模型很会『发现』根本不存在的规律。金融数据信噪比极低，这事每天发生。\n")

    # 杀手② 信号埋在噪声里
    ax = axes[0, 1]
    x = RNG.standard_normal(800)
    y_fin = 0.1*x + RNG.standard_normal(800)     # 微弱信号 + 大噪声
    ax.scatter(x, y_fin, s=6, alpha=0.3, color="#1f77b4")
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, 0.1*xs, "r-", lw=2, label="true signal (slope 0.1)")
    ax.set_title(f"(2) Finance: signal buried in noise (R^2={r2(y_fin,0.1*x):.3f})")
    ax.set_xlabel("feature"); ax.set_ylabel("future return")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"② 信号埋在噪声里：金融里一个真信号的可解释方差 R²≈{r2(y_fin,0.1*x):.3f}")
    print("   而识别猫的图像任务 R²≈0.95。差了近百倍——这是金融的『天花板』极低。\n")

    # 杀手③ 非平稳/机制转换
    ax = axes[1, 0]
    xr = RNG.standard_normal(300)
    y_reg1 = xr + 0.5*RNG.standard_normal(300)   # 机制1：正相关
    y_reg2 = -xr + 0.5*RNG.standard_normal(300)  # 机制2：反转!
    w = np.polyfit(xr, y_reg1, 1)                # 在机制1上训练
    oos = r2(y_reg2, np.polyval(w, xr))          # 拿到机制2上测
    ax.scatter(xr, y_reg1, s=8, alpha=0.4, color="#2ca02c", label="regime 1 (train)")
    ax.scatter(xr, y_reg2, s=8, alpha=0.4, color="#d62728", label="regime 2 (test, reversed)")
    ax.plot(xr, np.polyval(w, xr), "k-", lw=1.5, label="model from regime 1")
    ax.set_title(f"(3) Non-stationarity: rule reverses -> OOS R^2={oos:.1f}")
    ax.set_xlabel("feature"); ax.set_ylabel("return")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"③ 非平稳：训练时正相关的规律，测试时反转(机制转换)→ 样本外R²={oos:.1f}(惨负)。")
    print("   ML假设训练/测试同分布，金融恰恰违反它(闭环时间序列课)。\n")

    # 陷阱 交叉验证泄漏
    ax = axes[1, 1]
    N = 200
    t = np.arange(N)
    y_ac = np.cumsum(RNG.standard_normal(N)); y_ac = (y_ac - y_ac.mean())/y_ac.std()  # 自相关序列
    # 打乱k折：测试点的时间邻居混进了训练集
    perm = RNG.permutation(N); tr, te = perm[:140], perm[140:]
    r2_shuf = r2(y_ac[te], knn_time(t[tr], y_ac[tr], t[te]))
    # 时间顺序：训练在前,测试在后(诚实)
    r2_fwd = r2(y_ac[140:], knn_time(t[:140], y_ac[:140], t[140:]))
    ax.bar(["shuffled k-fold\n(LEAKS)", "time-ordered\n(honest)"], [r2_shuf, max(r2_fwd, -1)],
           color=["#d62728", "#2ca02c"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("(4) CV leakage: shuffling time series fakes high score")
    ax.set_ylabel("R^2"); ax.grid(alpha=0.3, axis="y")
    print(f"陷阱 交叉验证泄漏：自相关序列上，")
    print(f"   打乱k折 R²={r2_shuf:.2f}(虚高,因测试点的时间邻居漏进了训练)；")
    print(f"   时间顺序 R²={r2_fwd:.2f}(诚实)。→ 时序必须按时间切分,还要 purge/embargo。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "D2配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（噪声过拟合/低SNR/非平稳/CV泄漏 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
