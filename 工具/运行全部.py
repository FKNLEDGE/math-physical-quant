"""
一键跑通全部脚本 + 生成所有配图
================================
自动找到 量化/ 和 基础/ 下所有课程脚本，逐个运行，报告通过/失败，
并在各课目录生成对应配图。适合：①验证环境 ②一次性生成所有图。

运行（在项目根目录）：python 工具/运行全部.py
可选：python 工具/运行全部.py --quick   # 跳过较慢的脚本
"""

import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK = "--quick" in sys.argv
SLOW = {"honest_ml_strategy.py", "self_organized_criticality.py", "anomalous_diffusion.py",
        "crashes_phase_transition.py", "tree_models.py"}  # 较慢的


def find_scripts():
    scripts = []
    for base in ["量化", "基础"]:
        for p in sorted((ROOT / base).rglob("*.py")):
            if p.name == "生成示例数据.py":
                continue  # 数据生成器单独处理
            scripts.append(p)
    return scripts


def main():
    # 先确保示例数据存在
    gen = ROOT / "量化" / "数据" / "生成示例数据.py"
    if gen.exists() and not (ROOT / "量化/数据/示例数据/示例股_日线.csv").exists():
        print("生成示例数据中…")
        subprocess.run([sys.executable, str(gen)], capture_output=True)

    scripts = find_scripts()
    print(f"找到 {len(scripts)} 个脚本{'（--quick 模式跳过慢脚本）' if QUICK else ''}\n" + "=" * 56)
    ok, failed, skipped = 0, [], 0
    t0 = time.time()
    for p in scripts:
        if QUICK and p.name in SLOW:
            print(f"  ⏭️  跳过(慢)  {p.relative_to(ROOT)}"); skipped += 1
            continue
        t = time.time()
        r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=180)
        dt = time.time() - t
        if r.returncode == 0:
            print(f"  ✅ {dt:5.1f}s  {p.relative_to(ROOT)}"); ok += 1
        else:
            print(f"  ❌ {dt:5.1f}s  {p.relative_to(ROOT)}")
            print("     " + (r.stderr.strip().splitlines() or ["(无错误输出)"])[-1])
            failed.append(p)
    print("=" * 56)
    print(f"通过 {ok} | 失败 {len(failed)} | 跳过 {skipped} | 总耗时 {time.time()-t0:.0f}s")
    if failed:
        print("失败脚本：" + ", ".join(str(p.relative_to(ROOT)) for p in failed))
        sys.exit(1)
    print("🎉 全部通过！所有配图已生成在各课目录下。")


if __name__ == "__main__":
    main()
