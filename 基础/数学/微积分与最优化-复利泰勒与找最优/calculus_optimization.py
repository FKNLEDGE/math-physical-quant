"""
微积分与最优化 · 复利、泰勒与找最优（四个实验）
================================================
微积分讲『变化与累积』，最优化讲『找最好』。量化里到处是它们：
导数=敏感度(希腊字母)、e=连续复利、泰勒=风险逼近、梯度下降=ML引擎。

  实验①  e 的诞生：连续复利的极限 (1+1/n)^n → e
  实验②  泰勒展开：用切线(一阶)+曲率(二阶)逼近 ln(1+x)，解释波动率拖累
  实验③  梯度下降：像小球滚下山，找到函数最低点（ML 的引擎）
  实验④  最小方差组合：用最优化求"风险最低的权重"(接线性代数)

运行：python calculus_optimization.py
图内文字用英文；讲解在终端与 笔记.md。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def exp1_e_from_compounding(ax):
    """实验①：连续复利极限 (1+1/n)^n → e。"""
    ns = np.array([1, 2, 3, 5, 10, 50, 100, 1000, 10000])
    vals = (1 + 1.0/ns)**ns
    ax.semilogx(ns, vals, "o-", color="#1f77b4")
    ax.axhline(np.e, color="red", ls="--", label=f"e = {np.e:.5f}")
    ax.set_title("(1) e is born: (1+1/n)^n -> e")
    ax.set_xlabel("compounding times per year (n)")
    ax.set_ylabel("end value of 1 yuan @ 100%/yr")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("① e 的诞生：1 元，年利 100%，一年复利 n 次")
    for n in [1, 2, 12, 365]:
        print(f"   复利 {n:>4} 次 → {(1+1/n)**n:.5f}")
    print(f"   复利无穷次(连续) → e = {np.e:.5f}")
    print("   🔗 连续复利 → 对数收益率 ln(终值/初值)。这就是第03课为何用 log 收益。\n")


def exp2_taylor(ax):
    """实验②：泰勒展开逼近 ln(1+x)。一阶=切线，二阶=加曲率。"""
    x = np.linspace(-0.6, 0.6, 200)
    f = np.log(1 + x)
    f1 = x                      # 一阶：ln(1+x) ≈ x
    f2 = x - x**2/2             # 二阶：ln(1+x) ≈ x - x²/2
    ax.plot(x, f, "k-", lw=2, label="f(x)=ln(1+x)")
    ax.plot(x, f1, "--", color="#ff7f0e", label="1st order: x")
    ax.plot(x, f2, "--", color="#2ca02c", label="2nd order: x - x^2/2")
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title("(2) Taylor: tangent + curvature approximate a curve")
    ax.set_xlabel("x (= simple return r)")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("② 泰勒展开：ln(1+x) ≈ x - x²/2 + ...")
    print("   一阶(切线)在 0 附近就贴合；二阶(加曲率)贴合范围更大。")
    print("   🔗 ln(1+r) ≈ r - r²/2：对数收益≈简单收益减去『半个平方』")
    print("      —— 这正是『涨10%跌10%不回本』的波动率拖累，和随机过程的 -½σ² 同源！\n")


def gradient_descent(grad, x0, lr=0.3, steps=25):
    """最朴素的梯度下降：每一步往『下坡方向(负梯度)』挪一点。"""
    path = [np.array(x0, dtype=float)]
    x = np.array(x0, dtype=float)
    for _ in range(steps):
        x = x - lr * grad(x)
        path.append(x.copy())
    return np.array(path)


def exp3_gradient_descent(ax):
    """实验③：在 2D『碗』上做梯度下降，看它滚到谷底。"""
    # 目标函数 f(x,y) = x² + 0.2 y²（一个椭圆碗），梯度 = (2x, 0.4y)
    f = lambda p: p[0]**2 + 0.2*p[1]**2
    grad = lambda p: np.array([2*p[0], 0.4*p[1]])
    path = gradient_descent(grad, x0=[3.5, 3.5], lr=0.35, steps=20)

    xs = np.linspace(-4, 4, 100); ys = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(xs, ys)
    Z = X**2 + 0.2*Y**2
    ax.contour(X, Y, Z, levels=15, cmap="Greys", alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], "o-", color="#d62728", ms=3, label="descent path")
    ax.scatter([0], [0], color="green", marker="*", s=150, zorder=5, label="minimum")
    ax.set_title("(3) Gradient descent: roll downhill to the minimum")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("③ 梯度下降：从 (3.5, 3.5) 出发，每步沿『负梯度』下坡")
    print(f"   {len(path)-1} 步后到达 ({path[-1,0]:.3f}, {path[-1,1]:.3f})，逼近真最低点 (0,0)")
    print("   —— 这套『看坡度、往下挪』就是训练几乎所有机器学习模型的引擎。\n")


def exp4_min_variance(ax):
    """实验④：两资产最小方差组合——最优化的金融应用，接线性代数。"""
    sigA, sigB, rho = 0.18, 0.30, 0.20
    var = lambda w: w**2*sigA**2 + (1-w)**2*sigB**2 + 2*w*(1-w)*rho*sigA*sigB
    # 解析解（对 w 求导=0）：
    w_star = (sigB**2 - rho*sigA*sigB) / (sigA**2 + sigB**2 - 2*rho*sigA*sigB)
    # 数值解（梯度下降，把约束 0≤w≤1 简单截断）
    dvar = lambda w: 2*w*sigA**2 - 2*(1-w)*sigB**2 + 2*(1-2*w)*rho*sigA*sigB
    w = 0.0
    for _ in range(200):
        w = np.clip(w - 0.5*dvar(w), 0, 1)

    ws = np.linspace(0, 1, 200)
    ax.plot(ws, np.sqrt([var(x) for x in ws]), color="#1f77b4", label="portfolio vol")
    ax.scatter([w_star], [np.sqrt(var(w_star))], color="red", zorder=5,
               label=f"min @ w*={w_star:.2f}")
    ax.set_title("(4) Min-variance portfolio = optimization in finance")
    ax.set_xlabel("weight on asset A")
    ax.set_ylabel("portfolio volatility")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    print("④ 最小方差组合（A:18%波动, B:30%波动, 相关0.2）")
    print(f"   解析解 w* = {w_star:.4f}（最优配 A 的比例）")
    print(f"   数值解(梯度下降) w = {w:.4f}  （两者吻合 ✅）")
    print(f"   最低组合波动 = {np.sqrt(var(w_star)):.2%}，比单买波动更低的 A({sigA:.0%}) 还低！")
    print("   —— 这就是路线C『有效前沿』的起点：用最优化在风险-收益间找最好。\n")


def main():
    print("=" * 62)
    print("  微积分与最优化 · 复利、泰勒与找最优")
    print("=" * 62)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    exp1_e_from_compounding(axes[0, 0])
    exp2_taylor(axes[0, 1])
    exp3_gradient_descent(axes[1, 0])
    exp4_min_variance(axes[1, 1])
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "微积分与最优化配图.png"
    fig.savefig(out, dpi=110)
    print("=" * 62)
    print(f"  图已保存: {out.name}（e/泰勒/梯度下降/最小方差组合 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
