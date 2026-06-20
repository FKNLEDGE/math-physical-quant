"""
C2 · CAPM 与 Beta（只有系统性风险被定价）
=========================================
承接 C1：若人人持有市场组合，均衡下一只资产的预期收益只由它对市场的
敏感度(Beta)决定。这就是 CAPM。

  实验①  Beta = 资产收益对市场收益回归的『斜率』（高Beta vs 低Beta）
  实验②  证券市场线(SML)：预期超额收益 与 Beta 成正比
  实验③  系统性 vs 特异风险：分散能消掉特异风险，消不掉市场风险(地板)
  实验④  Alpha：Beta 解释不了的超额收益——是真本事还是运气？

运行：python capm_beta.py
依赖：numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

RNG = np.random.default_rng(7)
TD = 252
RF = 0.03                       # 无风险年利率
MKT_PREMIUM = 0.06              # 市场风险溢价(年)
MKT_VOL = 0.16                  # 市场波动(年)


def sim_market(n_days):
    """市场的【超额】日收益。"""
    return RNG.normal(MKT_PREMIUM/TD, MKT_VOL/np.sqrt(TD), n_days)


def sim_asset(mkt_ex, beta, idio_vol=0.20, alpha=0.0):
    """资产超额日收益 = alpha + beta·市场 + 特异噪声。"""
    eps = RNG.normal(alpha/TD, idio_vol/np.sqrt(TD), len(mkt_ex))
    return beta*mkt_ex + eps


def main():
    print("=" * 62)
    print("  C2 · CAPM 与 Beta")
    print("=" * 62)
    print("  逻辑：C1 说人人持有『无风险 + 同一切点组合』。若人人如此，")
    print("  那个切点组合必然就是【市场组合】(所有人持仓加总=整个市场)。")
    print("  均衡 → CAPM：  E[R_i] - rf = β_i · (E[R_m] - rf)")
    print("  β_i = Cov(R_i, R_m)/Var(R_m) = 资产对市场的敏感度 = 系统性风险\n")

    mkt = sim_market(2000)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① Beta = 回归斜率
    ax = axes[0, 0]
    for beta, c, name in [(1.5, "#d62728", "high beta=1.5"), (0.5, "#1f77b4", "low beta=0.5")]:
        r = sim_asset(mkt, beta)
        b_hat = np.cov(r, mkt)[0, 1]/np.var(mkt)
        ax.scatter(mkt*100, r*100, s=4, alpha=0.2, color=c)
        xs = np.array([mkt.min(), mkt.max()])*100
        ax.plot(xs, b_hat*xs, color=c, lw=2, label=f"{name} (est {b_hat:.2f})")
    ax.set_title("(1) Beta = slope of asset vs market returns")
    ax.set_xlabel("market excess return (%/day)"); ax.set_ylabel("asset excess return (%/day)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("① Beta = 资产对市场回归的斜率：高Beta放大市场波动，低Beta更防守。\n")

    # 图② 证券市场线 SML
    ax = axes[0, 1]
    betas = np.array([0.0, 0.4, 0.8, 1.0, 1.3, 1.6])
    emp_ret = []
    for b in betas:
        # 每个 beta 用一篮分散资产的平均，消掉特异噪声 → 点贴近理论 SML
        basket = np.mean([sim_asset(mkt, b) for _ in range(40)], axis=0)
        emp_ret.append(basket.mean()*TD)       # 年化超额收益
    ax.scatter(betas, np.array(emp_ret)*100, color="#1f77b4", s=50, zorder=5, label="diversified baskets")
    bx = np.linspace(0, 1.7, 50)
    ax.plot(bx, MKT_PREMIUM*bx*100, "r-", lw=2, label="SML: premium = beta x mkt premium")
    ax.set_title("(2) Security Market Line: return is linear in beta")
    ax.set_xlabel("beta"); ax.set_ylabel("expected excess return (%/yr)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("② 证券市场线：预期超额收益 = β × 市场风险溢价。Beta 越高，预期收益越高。\n")

    # 图③ 系统性 vs 特异风险：分散到地板
    ax = axes[1, 0]
    Ns = np.arange(1, 51)
    vols = []
    for N in Ns:
        # N 只 beta≈1 的股票等权组合：特异风险被分散，系统性风险留下
        rs = np.array([sim_asset(mkt, 1.0, idio_vol=0.30) for _ in range(N)])
        port = rs.mean(axis=0)
        vols.append(port.std()*np.sqrt(TD))
    ax.plot(Ns, np.array(vols)*100, "o-", color="#2ca02c", ms=3)
    ax.axhline(MKT_VOL*100, color="red", ls="--", label=f"systematic floor (mkt vol {MKT_VOL:.0%})")
    ax.set_title("(3) Diversification kills idiosyncratic, not market risk")
    ax.set_xlabel("number of stocks N"); ax.set_ylabel("portfolio vol (%/yr)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("③ 分散：把很多股票放一起，【特异风险】被平掉，只剩【系统性(市场)风险】地板。")
    print("   CAPM 的核心洞见：特异风险能免费分散掉 → 市场不为它付钱 → 只有 Beta 被定价。\n")

    # 图④ Alpha：Beta 解释不了的部分，是本事还是运气？
    ax = axes[1, 1]
    true_alphas = np.concatenate([np.zeros(40), [0.05, 0.08]])  # 42只:40只无alpha,2只有
    est_alphas = []
    for a in true_alphas:
        r = sim_asset(mkt, 1.0, idio_vol=0.25, alpha=a)
        b_hat = np.cov(r, mkt)[0, 1]/np.var(mkt)
        est_alphas.append((r.mean() - b_hat*mkt.mean())*TD)     # 回归截距=年化alpha
    est_alphas = np.array(est_alphas)*100
    ax.hist(est_alphas[:40], bins=15, color="#888", alpha=0.7, label="true alpha=0 (luck)")
    ax.axvline(est_alphas[40], color="#d62728", lw=2, label=f"true +5% -> est {est_alphas[40]:.1f}%")
    ax.axvline(est_alphas[41], color="#2ca02c", lw=2, label=f"true +8% -> est {est_alphas[41]:.1f}%")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("(4) Alpha: skill or luck? (zero-alpha funds scatter widely)")
    ax.set_xlabel("estimated alpha (%/yr)"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("④ Alpha = 回归截距 = Beta 解释不了的超额。40 只真实 alpha=0 的基金，")
    print(f"   估出来却散在 [{est_alphas[:40].min():.1f}%, {est_alphas[:40].max():.1f}%]——纯运气也能造出假Alpha。")
    print("   真有 5%/8% 本事的，也淹没在噪声里。呼应概率课：区分真Alpha和运气，需要很多年数据。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "C2配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（Beta回归/证券市场线/分散地板/Alpha 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
