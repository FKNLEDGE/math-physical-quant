"""
进阶·物理味 ① 随机矩阵理论清洗协方差
====================================
C4 说样本协方差噪声大、优化器会放大它。随机矩阵理论(RMT)给出物理学家的
principled 修法:它精确告诉你【哪些特征值是噪声】(Marchenko-Pastur 定律预测),
把噪声滤掉,得到稳健得多的协方差→更好的组合。这是核物理的数学(Wigner)用到金融。

  实验①  纯噪声的相关矩阵特征值,精确服从 Marchenko-Pastur 分布
  实验②  含真实结构(市场+行业因子)时:MP噪声块 + 几个'探出头'的信号特征值
  实验③  清洗:把噪声特征值压平(eigenvalue clipping),保留信号
  实验④  样本外:RMT清洗的协方差,做出的最小方差组合风险更低更稳(对比C4)

运行：python random_matrix_covariance.py
依赖：numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(0)


def mp_density(lam, q):
    """Marchenko-Pastur 密度:N个资产、T个观测的纯噪声相关矩阵特征值的理论分布。q=N/T。"""
    lm, lp = (1-np.sqrt(q))**2, (1+np.sqrt(q))**2
    out = np.zeros_like(lam)
    m = (lam > lm) & (lam < lp)
    out[m] = np.sqrt((lp-lam[m])*(lam[m]-lm)) / (2*np.pi*q*lam[m])
    return out, lm, lp


def structured_returns(T, N, n_sector=4):
    """市场因子 + 行业因子 + 特异噪声 → 有真实结构的收益。"""
    market = RNG.standard_normal(T)
    betas = RNG.uniform(0.4, 1.2, N)
    r = np.outer(market, betas)
    sectors = np.array_split(np.arange(N), n_sector)
    for s in sectors:
        f = 0.6*RNG.standard_normal(T)
        r[:, s] += np.outer(f, RNG.uniform(0.5, 1.0, len(s)))
    r += RNG.standard_normal((T, N))            # 特异噪声
    return r * 0.01                             # 缩放到真实日收益量级(~1%);相关矩阵尺度不变


def rmt_clean(corr, q):
    """RMT 清洗(eigenvalue clipping):把 < λ+ 的噪声特征值压成它们的平均,保留信号。"""
    vals, vecs = np.linalg.eigh(corr)
    _, _, lp = mp_density(np.array([1.0]), q)
    noise = vals < lp
    vals_c = vals.copy()
    if noise.sum() > 0:
        vals_c[noise] = vals[noise].mean()
    clean = vecs @ np.diag(vals_c) @ vecs.T
    d = np.sqrt(np.diag(clean))
    return clean / np.outer(d, d)               # 重新归一化对角为1


def min_var_weights(cov):
    inv = np.linalg.pinv(cov); ones = np.ones(len(cov))
    return inv @ ones / (ones @ inv @ ones)


def main():
    print("=" * 60)
    print("  进阶·物理味 ① 随机矩阵理论清洗协方差")
    print("=" * 60)
    N, T = 100, 500
    q = N/T
    print(f"  N={N} 资产, T={T} 观测, q=N/T={q:.2f}")
    _, lm, lp = mp_density(np.array([1.0]), q)
    print(f"  Marchenko-Pastur 噪声边界: λ-={lm:.2f}, λ+={lp:.2f}")
    print(f"  → 特征值在[{lm:.2f},{lp:.2f}]内=噪声(无法与随机区分); >λ+ 才是真信号。\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 图① 纯噪声 → MP 分布
    ax = axes[0, 0]
    noise_r = RNG.standard_normal((T, N))
    corr_noise = np.corrcoef(noise_r, rowvar=False)
    ev_noise = np.linalg.eigvalsh(corr_noise)
    ax.hist(ev_noise, bins=40, density=True, color="#1f77b4", alpha=0.7, label="noise eigenvalues")
    xs = np.linspace(lm*0.8, lp*1.1, 300)
    ax.plot(xs, mp_density(xs, q)[0], "r-", lw=2, label="Marchenko-Pastur theory")
    ax.set_title("(1) Pure noise: eigenvalues fit Marchenko-Pastur")
    ax.set_xlabel("eigenvalue λ"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("① 纯噪声相关矩阵的特征值,严丝合缝落在 MP 理论曲线下——噪声有【已知形状】。\n")

    # 图② 含结构 → MP块 + 信号探头
    ax = axes[0, 1]
    r = structured_returns(T, N)
    corr = np.corrcoef(r, rowvar=False)
    ev = np.linalg.eigvalsh(corr)
    n_signal = int((ev > lp).sum())
    ax.hist(ev[ev < lp*1.5], bins=40, density=True, color="#2ca02c", alpha=0.6, label="bulk (noise)")
    ax.plot(xs, mp_density(xs, q)[0], "r-", lw=2, label="MP (noise) bound")
    ax.axvline(lp, color="k", ls="--", label=f"λ+={lp:.2f}")
    for e in ev[ev > lp]:
        ax.axvline(e, color="#d62728", lw=1, alpha=0.6)
    ax.set_title(f"(2) Real structure: {n_signal} signal eigenvalues poke out")
    ax.set_xlabel("eigenvalue λ"); ax.set_ylabel("density")
    ax.set_xlim(0, lp*2.5); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"② 含市场+行业因子时:大部分特征值仍在MP噪声块里,但有 {n_signal} 个【探出 λ+】")
    print(f"   = 真实因子(最大的那个≈市场因子,{ev.max():.0f})。RMT 一眼分清信号与噪声。\n")

    # 图③ 清洗前后特征值
    ax = axes[1, 0]
    corr_clean = rmt_clean(corr, q)
    ev_clean = np.linalg.eigvalsh(corr_clean)
    ax.plot(sorted(ev, reverse=True), "o-", ms=3, color="#d62728", label="raw (noisy bulk)")
    ax.plot(sorted(ev_clean, reverse=True), "s-", ms=3, color="#1f77b4", label="RMT-cleaned (bulk flattened)")
    ax.axhline(lp, color="k", ls="--", lw=0.8, label=f"λ+={lp:.2f}")
    ax.set_yscale("log")
    ax.set_title("(3) Cleaning: flatten the noise bulk, keep signal")
    ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("eigenvalue (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print("③ 清洗:把噪声特征值压成平均(eigenvalue clipping),只留探出头的信号特征值。\n")

    # 图④ 样本外:RMT清洗组合风险更低
    ax = axes[1, 1]
    raw_risks, clean_risks = [], []
    for _ in range(60):
        data = structured_returns(2*T, N)
        ins, oos = data[:T], data[T:]
        std = ins.std(0)
        c_raw = np.corrcoef(ins, rowvar=False)
        c_cln = rmt_clean(c_raw, q)
        cov_raw = c_raw * np.outer(std, std)
        cov_cln = c_cln * np.outer(std, std)
        for cov, store in [(cov_raw, raw_risks), (cov_cln, clean_risks)]:
            w = min_var_weights(cov)
            store.append((oos @ w).std()*np.sqrt(252))   # 样本外实际波动(年化)
    ax.hist(raw_risks, bins=25, alpha=0.6, color="#d62728", label=f"raw cov (med {np.median(raw_risks):.1%})")
    ax.hist(clean_risks, bins=25, alpha=0.6, color="#1f77b4", label=f"RMT-cleaned (med {np.median(clean_risks):.1%})")
    ax.set_title("(4) OOS min-variance risk: cleaned is lower & tighter")
    ax.set_xlabel("realized OOS volatility"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"④ 样本外最小方差组合实际波动(60次中位数): 原始 {np.median(raw_risks):.1%} vs RMT清洗 {np.median(clean_risks):.1%}")
    print("   → 清洗后协方差更稳健,做出的组合样本外风险更低、更可控(直接改善C4的痛点)。\n")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "随机矩阵配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（MP分布/信号探头/清洗/样本外风险 四合一）")
    print("=" * 60)


if __name__ == "__main__":
    main()
