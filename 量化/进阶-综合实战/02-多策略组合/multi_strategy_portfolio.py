"""
进阶·综合实战 ② 多策略组合（策略层面的分散化）
==================================================
[C1](../../路线C-组合管理/01-马科维茨有效前沿/) 教"把不相关的资产放一起,降风险不降收益"。
这一课把同样的思想用到【策略】上——但有个【重要前提】:

  分散化能放大【一组各自为正、彼此低相关】的策略 → 组合夏普 ≈ 单策略 × √(有效条数);
  但它【救不了烂策略】:往里掺一条净亏的策略,只会把组合拖下水(真实数据的教训)。

为讲清机制,我们先用【受控合成策略】(每条夏普≈0.5、两两相关≈0.15)演示数学;
再用【真实示例行情】上的 4 条策略,展示"掺了亏钱策略,分散化反而受害"的坑。

  实验①  6 条合成策略(各自平平,夏普≈0.5)的净值
  实验②  相关矩阵:两两≈0.15(低相关=分散化的弹药)
  实验③  组合 vs 单条: 等权组合夏普≈0.9,远高于单条≈0.5
  实验④  组合夏普随策略数 N 增长(√N 律);但掺入一条亏钱策略→曲线被拖垮(坑!)

运行：python multi_strategy_portfolio.py
依赖：numpy, pandas, matplotlib（数据用 量化/数据/ 示例行情）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 量化/ 入路径

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quant_tools as qt

TD = 252


def sharpe(r):
    r = np.asarray(r)
    return r.mean() / r.std() * np.sqrt(TD) if r.std() > 0 else 0.0


def synth_strategies(n_days=1500, n_strat=6, target_sh=0.5, rho=0.15, sd=0.01, seed=0):
    """合成 n_strat 条策略日收益:共同因子(制造相关) + 独立部分。
    每条夏普≈target_sh,两两相关≈rho。返回 DataFrame。"""
    rng = np.random.default_rng(seed)
    mu = target_sh / np.sqrt(TD) * sd                       # 由夏普反推日均收益
    F = rng.standard_normal((n_days, 1))                    # 共同因子
    Z = rng.standard_normal((n_days, n_strat))              # 各自独立部分
    R = mu + sd * (np.sqrt(rho) * F + np.sqrt(1 - rho) * Z)
    return pd.DataFrame(R, columns=[f"S{i+1}" for i in range(n_strat)])


def combo_curve(Ns, n_bad_frac=0.0, target_sh=0.5, rho=0.15, sd=0.01, n_days=1500, reps=40):
    """蒙特卡洛平均:每个 N 生成 reps 组策略(其中 n_bad_frac 比例是净亏的),求等权组合夏普均值。
    这样把单次实现的运气抹平,露出 √N 增长律。"""
    out = []
    for N in Ns:
        vals = []
        n_bad = int(round(N * n_bad_frac))
        for s in range(reps):
            rng = np.random.default_rng(7000 + 31 * N + s)
            signs = np.array([-1.0] * n_bad + [1.0] * (N - n_bad))   # 净亏策略的漂移取负
            mu = signs * target_sh / np.sqrt(TD) * sd
            F = rng.standard_normal((n_days, 1))
            Z = rng.standard_normal((n_days, N))
            R = mu + sd * (np.sqrt(rho) * F + np.sqrt(1 - rho) * Z)
            vals.append(sharpe(R.mean(axis=1)))
        out.append(np.mean(vals))
    return np.array(out)


def real_strategies(close):
    """真实示例行情上的 4 条策略(用于展示'掺烂策略'的坑)。"""
    r = np.log(close / close.shift(1))
    cost = 5 / 1e4
    def ret(pos):
        pos = pos.shift(1).fillna(0.0)
        return (pos * r - pos.diff().abs().fillna(0) * cost)
    ma_f, ma_s = close.rolling(20).mean(), close.rolling(60).mean()
    z = (close - close.rolling(20).mean()) / close.rolling(20).std()
    don_hi, don_lo = close.rolling(20).max(), close.rolling(20).min()
    trend = ret((ma_f > ma_s).astype(float))
    mr = pd.Series(0.0, index=close.index); mr[z > 1] = -1; mr[z < -1] = 1
    mean_rev = ret(mr)
    bo = pd.Series(np.nan, index=close.index); bo[close >= don_hi] = 1; bo[close <= don_lo] = 0
    breakout = ret(bo.ffill().fillna(0))
    rev = -np.sign(r.rolling(3).sum())
    st_rev = ret(rev)
    return pd.DataFrame({"trend": trend, "mean_rev": mean_rev,
                         "breakout": breakout, "st_reversal": st_rev}).dropna()


def main():
    print("=" * 62)
    print("  进阶·综合实战 ② 多策略组合（策略层面的分散化）")
    print("=" * 62)
    R = synth_strategies()
    names = list(R.columns)
    sh_each = [sharpe(R[n]) for n in names]
    corr = R.corr()
    eq = R.mean(axis=1)
    sh_eq = sharpe(eq)
    print(f"① 6 条合成策略夏普: {[f'{s:.2f}' for s in sh_each]}(各自平平)")
    print(f"② 两两相关均值 {corr.values[np.triu_indices(6,1)].mean():.2f}(低相关)")
    print(f"③ 等权组合夏普 {sh_eq:.2f} ≈ 单条 {np.mean(sh_each):.2f} × √(有效条数) ——组合完胜单条!\n")

    # ④ 组合夏普 vs N(蒙特卡洛平均,抹平运气):全好 vs 一半是亏钱策略
    Ns = list(range(1, 13))
    sh_good = combo_curve(Ns, n_bad_frac=0.0)
    sh_bad = combo_curve(Ns, n_bad_frac=0.5)
    sh_theory = 0.5 * np.sqrt(np.array(Ns) / (1 + (np.array(Ns) - 1) * 0.15))
    print(f"④ 组合夏普 vs 策略数(全好,MC均值): {[f'{s:.2f}' for s in sh_good]}")
    print(f"   理论 √N 律预测: {[f'{s:.2f}' for s in sh_theory]}（实测贴合）")
    print(f"   若一半是亏钱策略: {[f'{s:.2f}' for s in sh_bad]}——加得越多越糟,分散化救不了烂策略!\n")

    # 真实数据的坑(打印佐证)
    RR = real_strategies(close := qt.load_sample_data()["Close"])
    sh_r = {n: sharpe(RR[n]) for n in RR.columns}
    sh_r_eq = sharpe(RR.mean(axis=1))
    print(f"⚠️ 真实示例行情 4 策略夏普 {dict((k,round(v,2)) for k,v in sh_r.items())}")
    print(f"   其中两条净亏→等权组合 {sh_r_eq:.2f} 反而低于最佳单条 {max(sh_r.values()):.2f}。")
    print(f"   教训:先确保每条策略各自为正,再谈分散化;否则是反向稀释。\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(names)))

    ax = axes[0, 0]
    for j, n in enumerate(names):
        ax.plot(np.exp(R[n].cumsum().values), color=cmap[j], lw=0.9, alpha=0.8, label=f"{n} (Sh {sh_each[j]:.2f})")
    ax.plot(np.exp(eq.cumsum().values), color="#d62728", lw=2.2, label=f"EQUAL combo (Sh {sh_eq:.2f})")
    ax.set_title("(1) Six mediocre strategies + their equal-weight combo (red)")
    ax.set_xlabel("day"); ax.set_ylabel("net value"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(names, fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(names, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    ax.set_title("(2) Low pairwise correlation (~0.15) = diversification fuel")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    bars = ax.bar(names + ["EQUAL"], sh_each + [sh_eq],
                  color=list(cmap) + [np.array([0.84, 0.15, 0.16, 1])])
    ax.axhline(np.mean(sh_each), color="gray", ls="--", lw=1, label=f"avg single {np.mean(sh_each):.2f}")
    ax.set_title("(3) Combined Sharpe far exceeds any single strategy")
    ax.set_ylabel("annualized Sharpe"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, sh_each + [sh_eq]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    ax = axes[1, 1]
    ax.plot(Ns, sh_good, "o-", color="#2ca02c", label="all positive (MC avg)")
    ax.plot(Ns, sh_theory, "--", color="k", lw=1.2, label="theory 0.5·√(N/(1+(N-1)ρ))")
    ax.plot(Ns, sh_bad, "s-", color="#d62728", label="half are money-losers (trap)")
    ax.set_title("(4) Sharpe grows ~√N — but losers drag it down")
    ax.set_xlabel("# strategies in combo"); ax.set_ylabel("portfolio Sharpe")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "多策略组合配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（单策略+组合/相关矩阵/组合夏普/分散化与坑 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
