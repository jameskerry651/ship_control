"""Matplotlib 字体：按平台选择可显示中文与常见 Unicode 符号的字体。"""

from __future__ import annotations

from matplotlib import font_manager


# 按优先级尝试；Latin 与 CJK 混排时 matplotlib 会按字形回退到列表中后续字体
_SANS_SERIF_PRIORITY = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def configure_matplotlib_fonts() -> None:
    """设置 sans-serif 回退链，并避免负号在部分 CJK 字体下显示为方块。"""
    import matplotlib.pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    chain = [n for n in _SANS_SERIF_PRIORITY if n in names]
    if not chain:
        chain = ["DejaVu Sans"]

    plt.rcParams["font.sans-serif"] = chain
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
