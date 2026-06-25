"""
进阶·风险管理 ① VaR 与 CVaR（尾部风险的行业标准度量）
========================================================
[03 风险语言](../../第03课-收益率与风险的语言/) 用波动率/夏普/回撤衡量风险。但风控台和监管(巴塞尔)
最常用的是 VaR 和 CVaR——专门盯【尾部】(会亏多惨)。这一课讲它们,以及它们的两个大坑。

  VaR(在险价值): X% 置信度下,一天最多亏多少(分位数)。例:99% 日 VaR=2.6% 指"99% 的日子亏不超 2.6%"。
  CVaR(条件VaR/期望损失 ES): 一旦突破 VaR,平均亏多少(尾部的均值)——更保守、且'相干'。

  两个大坑:
    ① 高斯 VaR 在【厚尾】下低估远端风险(99%+)——金融收益厚尾,用正态算 VaR 会自欺
    ② VaR【不次可加】:能惩罚分散化、不相干;CVaR 次可加、相干——这是 CVaR 胜出的根本原因

  实验①  收益分布 + 标出 VaR/CVaR(尾部在哪、CVaR 比 VaR 更深)
  实验②  历史 vs 高斯 VaR:99%/99.5% 远端,高斯系统性低估(厚尾的代价)
  实验③  VaR 回测:滚动 99% VaR 的突破次数——高斯 VaR 被突破得比该有的多
  实验④  次可加性:VaR(A+B) > VaR(A)+VaR(B)(违反!惩罚分散);CVaR 满足(奖励分散)

运行：python var_cvar.py
依赖：numpy, pandas, scipy, matplotlib（数据用 量化/数据/ 示例行情）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 量化/ 入路径

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import quant_tools as qt


def hist_var(x, a):
    return -np.quantile(x, a)


def gauss_var(x, a):
    return -(np.mean(x) + stats.norm.ppf(a) * np.std(x))


def cvar(x, a):
    """期望损失 = 最差 α 比例的平均损失(相干风险度量)。"""
    k = max(int(a * len(x)), 1)
    return -np.sort(x)[:k].mean()


def main():
    print("=" * 60)
    print("  进阶·风险管理 ① VaR 与 CVaR（尾部风险度量）")
    print("=" * 60)
    r = qt.to_log_returns(qt.load_sample_data()["Close"]).values
    v95, v99 = hist_var(r, 0.05), hist_var(r, 0.01)
    c95, c99 = cvar(r, 0.05), cvar(r, 0.01)
    print(f"  示例行情日收益:{len(r)} 天")
    print(f"① 95% VaR={v95*100:.2f}% (CVaR={c95*100:.2f}%) | 99% VaR={v99*100:.2f}% (CVaR={c99*100:.2f}%)")
    print(f"   CVaR>VaR:一旦突破 VaR,平均还要更深(尾部均值)。\n")

    # ② 历史 vs 高斯
    levels = [0.05, 0.01, 0.005]
    names = ["95%", "99%", "99.5%"]
    hv = [hist_var(r, a) for a in levels]
    gv = [gauss_var(r, a) for a in levels]
    print("② 历史 vs 高斯 VaR:")
    for nm, h, g in zip(names, hv, gv):
        print(f"   {nm}: 历史 {h*100:.2f}% | 高斯 {g*100:.2f}%  ({'高斯低估✓' if g < h else '高斯高估'})")
    print("   越往远端(99%+),高斯越低估——厚尾的代价(闭环 E1)。\n")

    # ③ 回测:滚动 99% VaR 的突破率
    win = 250
    breaches_g = breaches_h = total = 0
    bg, bh = [], []
    for t in range(win, len(r)):
        wr = r[t - win:t]
        if r[t] < -gauss_var(wr, 0.01):
            breaches_g += 1; bg.append(t)
        if r[t] < -hist_var(wr, 0.01):
            breaches_h += 1; bh.append(t)
        total += 1
    print(f"③ 99% VaR 回测(滚动{win}天,应突破≈1%):")
    print(f"   高斯 VaR 突破率 {breaches_g/total:.1%}（>1%=低估,被打脸更多）| 历史 VaR {breaches_h/total:.1%}\n")

    # ④ 次可加性(两只独立违约债)
    rng = np.random.default_rng(0)
    A = np.where(rng.random(400000) < 0.04, -1.0, 0.05)
    B = np.where(rng.random(400000) < 0.04, -1.0, 0.05)
    a = 0.05
    var_sum, var_AB = hist_var(A, a) + hist_var(B, a), hist_var(A + B, a)
    cv_sum, cv_AB = cvar(A, a) + cvar(B, a), cvar(A + B, a)
    print(f"④ 次可加性(95%, 两只独立违约债):")
    print(f"   VaR(A)+VaR(B)={var_sum:.2f} vs VaR(A+B)={var_AB:.2f} → {'违反!VaR惩罚分散化' if var_AB > var_sum else '满足'}")
    print(f"   CVaR(A)+CVaR(B)={cv_sum:.2f} vs CVaR(A+B)={cv_AB:.2f} → {'满足次可加,CVaR奖励分散化(相干)' if cv_AB <= cv_sum else '违反'}\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(r * 100, bins=80, color="#1f77b4", alpha=0.6)
    ax.axvline(-v99 * 100, color="#d62728", lw=2, label=f"99% VaR = {v99*100:.2f}%")
    ax.axvline(-c99 * 100, color="#7f0000", lw=2, ls="--", label=f"99% CVaR = {c99*100:.2f}%")
    tail = r[r <= -v99] * 100
    ax.hist(tail, bins=20, color="#d62728", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("(1) Return distribution: VaR (quantile) vs CVaR (tail mean)")
    ax.set_xlabel("daily return %"); ax.set_ylabel("count (log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, np.array(hv) * 100, w, color="#1f77b4", label="historical VaR")
    ax.bar(x + w / 2, np.array(gv) * 100, w, color="#d62728", label="Gaussian VaR")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("(2) Gaussian VaR underestimates the far tail (fat tails)")
    ax.set_ylabel("VaR %"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.annotate("Gaussian too low\nat 99%+", xy=(1.18, gv[1] * 100), xytext=(1.5, gv[1] * 100 * 1.3),
                fontsize=8, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))

    ax = axes[1, 0]
    idx = np.arange(win, len(r))
    ax.plot(idx, r[win:] * 100, color="#bbbbbb", lw=0.5, label="daily return")
    ax.scatter(bg, r[bg] * 100, s=18, color="#d62728", zorder=5, label=f"Gaussian-VaR breach ({breaches_g/total:.1%})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("(3) 99% VaR backtest: Gaussian breached too often (>1%)")
    ax.set_xlabel("day"); ax.set_ylabel("return %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar([0, 1], [var_sum, var_AB], width=0.5, color=["#888", "#d62728"])
    ax.bar([3, 4], [cv_sum, cv_AB], width=0.5, color=["#888", "#2ca02c"])
    ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["VaR\nA+B sep", "VaR\nA+B comb", "CVaR\nsep", "CVaR\ncomb"], fontsize=8)
    ax.set_title("(4) VaR not subadditive (punishes diversification); CVaR is coherent")
    ax.set_ylabel("risk (loss units)"); ax.grid(alpha=0.3, axis="y")
    ax.annotate("VaR↑ when\ncombined!", xy=(1, var_AB), xytext=(1.3, var_AB * 0.6),
                fontsize=8, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.annotate("CVaR↓\n(diversify)", xy=(4, cv_AB), xytext=(3.3, max(cv_sum, cv_AB) * 0.6),
                fontsize=8, color="#2ca02c", arrowprops=dict(arrowstyle="->", color="#2ca02c"))

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "VaR_CVaR配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（分布/高斯低估/回测/次可加 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
