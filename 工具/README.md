# 工具 🛠️

小工具，方便你用这个项目。

| 工具 | 作用 | 用法 |
|---|---|---|
| [运行全部.py](运行全部.py) | 一键跑通所有课程脚本、生成全部配图（也可验证环境） | `python 工具/运行全部.py`（加 `--quick` 跳过慢脚本） |
| [中文字体.py](中文字体.py) | 让 matplotlib 显示中文（可选） | 画图前 `from 工具.中文字体 import 启用中文; 启用中文()` |
| [生成Notebook.py](生成Notebook.py) | 把课程 `.py` 转成交互式 Jupyter `.ipynb`（小白更友好） | `python 工具/生成Notebook.py 某课/xxx.py`，或 `--all` 全转 |

### 📓 关于 Notebook 版

课程默认是 `.py + 笔记.md`（稳健、便于版本管理）。但 **Jupyter Notebook 对小白更友好**——可以一段段运行、即时看结果和图。已为几节关键课生成了示例 `.ipynb`（与 `.py` 同目录）：
- [量化/第01课/hello_quant.ipynb](../量化/第01课-量化是什么/)
- [量化/第04课/lesson04_ma_crossover.ipynb](../量化/第04课-第一个策略-均线交叉/)
- [量化/路线A+/02-端到端实战项目/end_to_end_project.ipynb](../量化/路线A+实战进阶/02-端到端实战项目/)

想给任意一课（或全部）生成 notebook，用 `生成Notebook.py` 即可。用法：先 `pip install jupyterlab`，再 `jupyter lab` 打开 `.ipynb`，从上往下逐格运行（Shift+Enter）。

---

## 关于图表为什么默认用英文

本项目所有配图的**标题/坐标轴用英文**，是为了**在任何电脑上都能干净渲染**（很多系统的 matplotlib 没配中文字体，中文会变成"方块□□□"）。**所有讲解都在 `笔记.md` 和终端输出里，是中文的**，不影响理解。

如果你想让自己实验的图显示中文，用 `中文字体.py` 即可（前提是你系统装了中文字体）。

> Linux 装中文字体：`sudo apt install fonts-noto-cjk` 或 `fonts-wqy-zenhei`。
