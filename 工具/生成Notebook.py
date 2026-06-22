"""
把课程 .py 自动转成 Jupyter Notebook (.ipynb)
==============================================
Notebook 对小白更友好:可以一段段运行、即时看结果和图。
本工具把任意课程脚本拆成单元格(docstring→说明, 每个函数→一格, 末尾运行),
并自动改成内联显示图表。无需额外依赖(直接生成 .ipynb 的 JSON)。

用法：
    python 工具/生成Notebook.py 量化/第01课-量化是什么/hello_quant.py
    # 在同目录生成 hello_quant.ipynb
    python 工具/生成Notebook.py --all     # 给所有课程脚本生成 notebook
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    text = text.replace('matplotlib.use("Agg")', '# matplotlib.use("Agg")  # notebook 内联显示，无需 Agg')
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def py_to_notebook(py_path: Path) -> dict:
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cells = []

    # 1) 模块 docstring → 开篇说明
    doc = ast.get_docstring(tree)
    title = py_path.stem
    intro = f"# {title}\n\n> 本 Notebook 由 `{py_path.name}` 自动生成，可逐格运行、即时看结果。\n"
    if doc:
        intro += "\n```\n" + doc + "\n```\n"
    cells.append(md_cell(intro))
    cells.append(code_cell("%matplotlib inline"))

    # 2) 顶层节点:imports/常量攒成一格,每个 def/class 一格,末尾 if-main 一格
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]   # 跳过 docstring
    buffer = []
    def flush():
        if buffer:
            cells.append(code_cell("\n".join(buffer)))
            buffer.clear()
    for node in body:
        seg = ast.get_source_segment(src, node)
        if seg is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush()
            cells.append(code_cell(seg))
        elif isinstance(node, ast.If):     # if __name__ == "__main__": main()
            flush()
            cells.append(md_cell("### ▶️ 运行\n下面这格会执行主流程并显示图表："))
            cells.append(code_cell(seg))
        else:
            buffer.append(seg)
    flush()

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def convert(py_path: Path):
    nb = py_to_notebook(py_path)
    out = py_path.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ✅ {py_path.relative_to(ROOT)}  ->  {out.name}")
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return
    if args[0] == "--all":
        scripts = [p for base in ["量化", "基础"] for p in sorted((ROOT/base).rglob("*.py"))
                   if p.name != "生成示例数据.py"]
        print(f"转换 {len(scripts)} 个脚本为 notebook：")
        for p in scripts:
            convert(p)
    else:
        for a in args:
            convert(Path(a).resolve())


if __name__ == "__main__":
    main()
