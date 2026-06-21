"""
让 matplotlib 显示中文（可选小工具）
====================================
本项目的配图为了"到处都能干净渲染"，图内文字默认用英文。
如果你想在自己的实验里让图表显示【中文】，在画图前调用一次本工具即可。

用法：
    from 工具.中文字体 import 启用中文
    启用中文()
    # 之后正常用 matplotlib，标题/标签可以写中文
或在任意脚本顶部复制下面 try 这几行。

直接运行可检测你系统有没有可用中文字体：
    python 工具/中文字体.py
"""

import matplotlib
import matplotlib.font_manager as fm

# 常见中文字体（Windows / Mac / Linux）
CANDIDATES = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
              "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
              "Noto Sans CJK JP", "Arial Unicode MS"]


def 启用中文(verbose=True):
    """找一个系统里可用的中文字体并配置给 matplotlib。返回字体名或 None。"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False   # 修负号显示
            if verbose:
                print(f"✅ 已启用中文字体：{name}")
            return name
    if verbose:
        print("⚠️ 没找到中文字体。图内中文会显示成方块。")
        print("   解决：装一个中文字体（如 Noto Sans CJK / 文泉驿），或图内继续用英文。")
        print("   Linux 可： sudo apt install fonts-noto-cjk  /  fonts-wqy-zenhei")
    return None


# 英文别名，方便代码里调用
use_chinese_font = 启用中文


if __name__ == "__main__":
    启用中文()
