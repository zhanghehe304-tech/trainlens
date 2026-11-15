"""trainlens.viz —— 把指标画成图，全部返回 base64 PNG，方便直接塞进 HTML。"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前设置，服务端/无桌面环境才不会炸

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

__all__ = ["line_chart", "confusion_heatmap", "error_grid"]

# 中文字体优先，缺失时回落 DejaVu；unicode_minus 关掉是为了负号不变豆腐块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PALETTE = ("#2b6cb0", "#d9480f", "#2f855a", "#805ad5", "#b7791f")
INK = "#1a202c"
MUTED = "#718096"
GRID = "#e6e8eb"


def _to_b64(fig, dpi: int = 120) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _soften(values: Sequence[float], alpha: float = 0.1) -> np.ndarray:
    """指数滑动平均，长序列画出来才看得清趋势。"""
    out: List[float] = []
    last: Optional[float] = None
    for v in values:
        last = float(v) if last is None else (1 - alpha) * last + alpha * float(v)
        out.append(last)
    return np.asarray(out, dtype=float)


def _dress(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def line_chart(
    x: Sequence[float],
    series: Dict[str, Sequence[float]],
    title: str,
    xlabel: str = "epoch",
    ylabel: str = "",
    logy: bool = False,
) -> str:
    """一条或多条曲线的折线图。序列长度需与 x 一致。"""
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    for i, (name, values) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        values = np.asarray(values, dtype=float)
        if values.size != len(x):
            raise ValueError(f"曲线 {name} 与横轴长度不一致: {values.size} vs {len(x)}")
        if values.size > 50:
            ax.plot(x, values, color=color, linewidth=0.8, alpha=0.25)
            ax.plot(x, _soften(values), color=color, linewidth=1.6, label=f"{name} (smooth)")
        else:
            ax.plot(x, values, color=color, linewidth=1.6, marker="o", markersize=3, label=name)

    if logy:
        ax.set_yscale("log")
    _dress(ax, title, xlabel, ylabel)
    if series:
        ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    return _to_b64(fig)


def confusion_heatmap(cm: np.ndarray, labels: Sequence[str]) -> str:
    """混淆矩阵热力图；类别多时自动省略格内数字。"""
    n = cm.shape[0]
    size = max(3.4, 0.42 * n + 1.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.92))

    image = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=8)

    annotate = n <= 14
    if annotate:
        threshold = cm.max() / 2 if cm.size else 0
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(
                        j, i, f"{cm[i, j]:d}", ha="center", va="center",
                        fontsize=max(6, 10 - n // 4),
                        color="white" if cm[i, j] > threshold else INK,
                    )

    _dress(ax, "Confusion matrix", "Predicted", "True")
    ax.grid(False)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    rotate = 45 if n > 8 else 0
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_yticklabels(labels)
    fig.tight_layout()
    return _to_b64(fig)


def error_grid(
    images: Sequence[np.ndarray],
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    ncols: int = 4,
) -> str:
    """错分样本缩略图，每个格子标注 真值 / 预测 / 置信度。"""
    count = len(images)
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.55, nrows * 1.72))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= count:
            continue
        image = np.asarray(images[i], dtype=float)
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(image, 0, 1))

        caption = f"T:{true_labels[i]}  P:{pred_labels[i]}"
        if scores is not None:
            caption += f"  {float(scores[i]):.2f}"
        ax.set_title(caption, fontsize=7.5, color=INK, pad=4)

    fig.tight_layout()
    return _to_b64(fig)
