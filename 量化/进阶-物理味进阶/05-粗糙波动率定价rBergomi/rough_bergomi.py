"""
进阶·物理味 ⑤ 粗糙波动率定价（rough Bergomi）
================================================
[④ 粗糙波动率](../04-粗糙波动率/) 测出"对数波动率像 H≈0.1 的粗糙过程",并【理论上】预言
ATM 偏斜的期限结构 ~ T^(H−1/2)。这一课把粗糙【搬进期权定价】,用模拟的期权价【验证】那条预言——
完成 ④ 的闭环,也是 [② Heston](../02-随机波动率Heston/) 的"粗糙升级版"。

用 rough Bergomi 模型(最干净的粗糙波动率定价模型,rough Heston 的同族;方差由粗糙
Volterra 过程驱动,但模拟更稳健、适合教学):

    V_t = ξ₀ · exp( η√(2H)·Y_t − ½η²t^(2H) ),  Y_t = ∫₀ᵗ (t−s)^(H−½) dW_s（粗糙)
    dS = √V · S · ( ρ dW + √(1−ρ²) dW⊥ )      （ρ<0 = 杠杆,生左偏)

  实验①  方差路径:H=0.1(粗糙锯齿) vs H=0.5(光滑),粗糙度一眼可见
  实验②  短到期(T=0.1)隐含波动率微笑:粗糙的偏斜【陡得多】
  实验③  ATM 偏斜期限结构:粗糙≈T^(H−½)爆炸(验证④预言),H=0.5 平坦(Heston硬伤)
  实验④  粗糙波动率的整张微笑面:短到期陡、长到期平——拟合真实期权面的关键

运行：python rough_bergomi.py
依赖：numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path

N = stats.norm.cdf


def bs_call(S, K, r, sig, T):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)


def implied_vol(price, S, K, r, T):
    try:
        return optimize.brentq(lambda s: bs_call(S, K, r, s, T) - price, 1e-3, 4.0)
    except ValueError:
        return np.nan


def rbergomi(H, eta, rho, xi0, T, n, npaths, rng, S0=100.0):
    """模拟 rough Bergomi 的价格路径与方差路径。返回 (S[npaths,n], V, 时间网格)。"""
    dt = T / n
    tt = np.arange(1, n + 1) * dt
    b = (np.arange(1, n + 1) * dt)**(H - 0.5) * np.sqrt(dt)   # Volterra 卷积核权重(lag k)
    Z = rng.standard_normal((npaths, n))                      # 驱动方差的布朗增量(标准化)
    Y = np.zeros((npaths, n))
    for i in range(1, n):
        Y[:, i] = Z[:, :i] @ b[i - 1::-1]                     # Y_i = Σ_{j<i} (t_i-t_j)^(H-½)√dt Z_j
    V = xi0 * np.exp(eta * np.sqrt(2 * H) * Y - 0.5 * eta**2 * tt**(2 * H))
    W = Z * np.sqrt(dt)                                        # 同一布朗→价格与波动相关(杠杆)
    Wp = rng.standard_normal((npaths, n)) * np.sqrt(dt)
    dlogS = -0.5 * V * dt + np.sqrt(V) * (rho * W + np.sqrt(1 - rho**2) * Wp)
    S = S0 * np.exp(np.cumsum(dlogS, axis=1))
    return S, V, tt


def smile(S_T, T, S0=100.0, r=0.0, strikes=None):
    """给定到期价样本,反解一条隐含波动率微笑(对一组行权价)。"""
    if strikes is None:
        strikes = np.linspace(85, 115, 13)
    ivs = []
    for K in strikes:
        price = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
        ivs.append(implied_vol(price, S0, K, r, T))
    return strikes, np.array(ivs)


def atm_skew(S_T, T, S0=100.0):
    """ATM 偏斜 = dIV/d(log K) 在平值附近的斜率(取绝对值便于看期限结构)。"""
    Ks = np.array([92.0, 96.0, 100.0, 104.0, 108.0])
    _, ivs = smile(S_T, T, S0, strikes=Ks)
    lk = np.log(Ks / S0)
    g = np.isfinite(ivs)
    return abs(np.polyfit(lk[g], ivs[g], 1)[0]) if g.sum() >= 2 else np.nan


def main():
    print("=" * 62)
    print("  进阶·物理味 ⑤ 粗糙波动率定价（rough Bergomi）")
    print("=" * 62)
    eta, rho, xi0, T, n, npaths = 1.5, -0.7, 0.04, 1.0, 200, 40000
    rng = np.random.default_rng(0)
    S_r, V_r, tt = rbergomi(0.1, eta, rho, xi0, T, n, npaths, np.random.default_rng(1))
    S_s, V_s, _ = rbergomi(0.5, eta, rho, xi0, T, n, npaths, np.random.default_rng(1))
    print(f"  参数: η={eta}(vol of vol), ρ={rho}(杠杆), ξ₀={xi0}(≈20%波动); 路径 {npaths}\n")

    print(f"① 方差路径: H=0.1 粗糙锯齿(标准差大、回归快) vs H=0.5 光滑\n")

    # ② 短到期微笑
    iT = int(0.1 * n) - 1
    Ks, iv_r = smile(S_r[:, iT], tt[iT])
    _, iv_s = smile(S_s[:, iT], tt[iT])
    sk_r0, sk_s0 = atm_skew(S_r[:, iT], tt[iT]), atm_skew(S_s[:, iT], tt[iT])
    print(f"② 短到期 T={tt[iT]:.2f} 的 ATM 偏斜: 粗糙 H=0.1 → {sk_r0:.2f} ≫ 光滑 H=0.5 → {sk_s0:.2f}")
    print("   粗糙波动率的短期微笑陡得多——正是真实期权市场的样子。\n")

    # ③ 偏斜期限结构
    mats_idx = [int(m * n) - 1 for m in [0.08, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]]
    Ts = tt[mats_idx]
    skew_r = np.array([atm_skew(S_r[:, i], tt[i]) for i in mats_idx])
    skew_s = np.array([atm_skew(S_s[:, i], tt[i]) for i in mats_idx])
    # ④ 理论幂律 T^(H-1/2),用最长到期标定
    power = Ts**(0.1 - 0.5)
    power = power / power[-1] * skew_r[-1]
    print(f"③ 偏斜期限结构(ATM skew vs T):")
    print(f"   粗糙 H=0.1: 从 {skew_r[0]:.2f}(短) 降到 {skew_r[-1]:.2f}(长)——随 T→0 爆炸,贴合 T^(H-½)")
    print(f"   光滑 H=0.5: {np.round(skew_s,2)} 基本平坦(Heston 给不出陡短偏斜)\n")

    # ==================== 画图 ====================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    ax = axes[0, 0]
    tg = np.arange(n) / n
    ax.plot(tg, np.sqrt(V_s[0]) * 100, color="#1f77b4", lw=0.9, label="H=0.5 smooth")
    ax.plot(tg, np.sqrt(V_r[0]) * 100, color="#d62728", lw=0.8, label="H=0.1 ROUGH")
    ax.set_title("(1) Instantaneous vol path: rough (H=0.1) is jagged")
    ax.set_xlabel("time"); ax.set_ylabel("volatility %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(Ks, iv_r * 100, "o-", color="#d62728", ms=3, label=f"rough H=0.1 (skew {sk_r0:.2f})")
    ax.plot(Ks, iv_s * 100, "s-", color="#1f77b4", ms=3, label=f"smooth H=0.5 (skew {sk_s0:.2f})")
    ax.axvline(100, color="gray", ls=":", lw=0.8)
    ax.set_title(f"(2) Short-maturity (T={tt[iT]:.2f}) implied-vol smile: rough is much steeper")
    ax.set_xlabel("strike K"); ax.set_ylabel("implied vol %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(Ts, skew_r, "o-", color="#d62728", label="rough H=0.1 (measured)")
    ax.plot(Ts, skew_s, "s-", color="#1f77b4", label="smooth H=0.5 (measured)")
    ax.plot(Ts, power, "--", color="k", lw=1.2, label="theory ~T^(H-½), H=0.1")
    ax.set_title("(3) ATM skew term structure: rough explodes short-term (validates ④)")
    ax.set_xlabel("maturity T (years)"); ax.set_ylabel("|ATM skew|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for m, c in [(0.08, "#d62728"), (0.25, "#ff7f0e"), (0.5, "#9467bd"), (1.0, "#1f77b4")]:
        i = int(m * n) - 1
        Kk, ivv = smile(S_r[:, i], tt[i])
        ax.plot(Kk, ivv * 100, "o-", ms=2.5, color=c, label=f"T={tt[i]:.2f}")
    ax.axvline(100, color="gray", ls=":", lw=0.8)
    ax.set_title("(4) Rough-vol smile surface: short steep, long flat")
    ax.set_xlabel("strike K"); ax.set_ylabel("implied vol %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "粗糙定价配图.png"
    fig.savefig(out, dpi=110)
    print(f"  图已保存: {out.name}（方差路径/短期微笑/偏斜期限/微笑面 四合一）")
    print("=" * 62)


if __name__ == "__main__":
    main()
