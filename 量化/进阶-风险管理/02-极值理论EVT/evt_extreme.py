"""
进阶·风险管理 ② 极值理论（EVT）与极端 VaR
==============================================
[① VaR/CVaR](../01-VaR与CVaR/) 留了个坑:到 99.9%、99.99% 这种【极端分位】,你几乎没有数据——
历史模拟没观测可用、高斯又把厚尾算没了。极值理论(EVT)给出 principled 的办法:【外推到没见过的极端】。

  核心定理(Pickands-Balkema-de Haan):**任何分布,超过一个高阈值的部分,都收敛到同一族——
  广义帕累托分布(GPD)。** 于是你只需把 GPD 拟合到尾巴,就能外推到任意极端分位。

  形状参数 ξ 决定尾巴有多重:ξ>0=厚尾(幂律,尾指数 1/ξ);ξ=0=指数(类高斯);ξ<0=有界。
  金融收益 ξ>0。

  实验①  超阈值(POT):取超过高阈值的损失,拟合 GPD
  实验②  尾部拟合(log-log):经验尾 vs GPD(EVT) vs 高斯——高斯衰减太快,EVT 贴合厚尾
  实验③  极端 VaR(99%→99.99%):高斯严重低估、历史外推不了、EVT 给出有理有据的极端估计
  实验④  平均超额函数:正斜率(线性)=厚尾的 GPD 签名,并读出形状参数 ξ

运行：python evt_extreme.py
依赖：numpy, scipy, matplotlib（数据用 量化/数据/ 示例行情）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 量化/ 入路径

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import quant_tools as qt


def main():
    print("=" * 60)
    print("  进阶·风险管理 ② 极值理论（EVT）与极端 VaR")
    print("=" * 60)
    r = qt.to_log_returns(qt.load_sample_data()["Close"]).values
    loss = -r
    n = len(loss)
    u = np.quantile(loss, 0.90)                 # 阈值=90 分位
    exc = loss[loss > u] - u                     # 超阈值部分
    Nu = len(exc)
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    print(f"  样本 {n} 天;阈值 u={u*100:.2f}%(90分位),超阈值 {Nu} 个")
    print(f"① GPD 拟合: 形状 ξ={xi:.3f}(>0=厚尾,尾指数≈{1/xi:.1f}), 尺度 β={beta*100:.2f}%\n")

    def evt_var(a):
        return u + (beta / xi) * (((n / Nu) * (1 - a)) ** (-xi) - 1)

    def evt_es(a):
        return (evt_var(a) + beta - xi * u) / (1 - xi)

    levels = [0.99, 0.995, 0.999, 0.9999]
    print("③ 极端 VaR 对比:")
    for a in levels:
        g = -(r.mean() + stats.norm.ppf(1 - a) * r.std())
        h = np.quantile(loss, a)
        beyond = "  ← 超出数据范围,历史不可信" if (1 - a) * n < 2 else ""
        print(f"   {a*100:5.2f}%: 高斯 {g*100:.2f}% | 历史 {h*100:.2f}% | EVT {evt_var(a)*100:.2f}%(ES {evt_es(a)*100:.2f}%){beyond}")
    print("   越极端,高斯越离谱、历史越没数据;EVT 靠拟合尾形,有理有据地外推。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① POT + GPD 密度
    ax = axes[0, 0]
    ax.hist(exc * 100, bins=30, density=True, color="#1f77b4", alpha=0.6, label="exceedances over u")
    xs = np.linspace(0, exc.max(), 200)
    ax.plot(xs * 100, stats.genpareto.pdf(xs, xi, 0, beta) / 100, "r-", lw=2, label=f"GPD fit (ξ={xi:.2f})")
    ax.set_title(f"(1) Peaks-over-threshold (u={u*100:.2f}%) fit by GPD")
    ax.set_xlabel("loss beyond threshold %"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 图② 尾部 CCDF log-log
    ax = axes[0, 1]
    s = np.sort(loss[loss > u])
    emp = 1 - np.arange(len(s)) / len(s)
    ax.loglog(s * 100, emp * (Nu / n), "o", ms=3, color="#1f77b4", label="empirical tail")
    gpd_ccdf = (Nu / n) * stats.genpareto.sf(s - u, xi, 0, beta)
    ax.loglog(s * 100, gpd_ccdf, "r-", lw=2, label="GPD (EVT)")
    gauss_ccdf = stats.norm.sf(s, r.mean(), r.std())
    ax.loglog(s * 100, gauss_ccdf, "--", color="#888", lw=1.5, label="Gaussian (too thin)")
    ax.set_title("(2) Tail CCDF: Gaussian decays too fast, EVT fits the fat tail")
    ax.set_xlabel("loss %"); ax.set_ylabel("P(loss ≥ x)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # 图③ 极端 VaR vs 置信度
    ax = axes[1, 0]
    aa = 1 - np.logspace(-2, -4, 30)
    g = np.array([-(r.mean() + stats.norm.ppf(1 - a) * r.std()) for a in aa])
    e = np.array([evt_var(a) for a in aa])
    ax.plot((1 - aa) * 100, g * 100, "--", color="#888", lw=2, label="Gaussian VaR (underestimates)")
    ax.plot((1 - aa) * 100, e * 100, "-", color="#d62728", lw=2, label="EVT VaR (extrapolates)")
    hx = [0.01, 0.005, 0.001]
    ax.plot([x * 100 for x in hx], [np.quantile(loss, 1 - x) * 100 for x in hx], "o",
            color="#1f77b4", ms=7, label="historical (where data exists)")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_title("(3) Extreme VaR: Gaussian too low, EVT extrapolates safely")
    ax.set_xlabel("tail probability % (→ more extreme)"); ax.set_ylabel("VaR %")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # 图④ 平均超额函数
    ax = axes[1, 1]
    us = np.quantile(loss, np.linspace(0.80, 0.98, 15))
    me = [loss[loss > uu].mean() - uu for uu in us]
    ax.plot(us * 100, np.array(me) * 100, "o-", color="#2ca02c")
    slope = np.polyfit(us, me, 1)[0]
    ax.set_title(f"(4) Mean-excess plot: positive slope = fat tail (ξ≈{slope/(1+slope):.2f})")
    ax.set_xlabel("threshold u %"); ax.set_ylabel("mean excess over u %")
    ax.grid(alpha=0.3)
    ax.annotate("upward slope\n= GPD fat tail", xy=(us[7] * 100, me[7] * 100),
                xytext=(us[2] * 100, max(me) * 100 * 1.05), fontsize=8.5, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c"))

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "EVT配图.png"
    fig.savefig(out, dpi=110)
    print(f"④ 平均超额函数斜率 {slope:.2f}>0 → 厚尾(GPD 适用)。")
    print(f"  图已保存: {out.name}（POT拟合/尾部CCDF/极端VaR/平均超额 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
