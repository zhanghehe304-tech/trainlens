"""trainlens.tracker —— 与框架无关的训练记录器。

它只做三件事：记标量指标、收验证集预测、留错分样本。
渲染成网页这一步交给 :meth:`Tracker.report`。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Tracker"]

_LABEL_KEYS = ("epoch", "step")


class Tracker:
    """一次训练运行的记录器。

    最小用法::

        tracker = Tracker("baseline", classes=["cat", "dog"])
        for epoch in range(10):
            for step, batch in enumerate(loader):
                ...
                tracker.log(epoch=epoch, step=step, loss=loss, acc=acc, lr=lr)
            tracker.end_epoch(val_loss=val_loss, val_acc=val_acc)
            tracker.add_predictions(y_val, pred, reset=True)
            tracker.add_errors(x_val, y_val, pred, scores=conf)
        tracker.report("reports/")
    """

    def __init__(
        self,
        name: str = "run",
        classes: Optional[Sequence[Any]] = None,
        notes: str = "",
    ) -> None:
        self.name = str(name)
        self.classes: Optional[List[str]] = (
            [str(c) for c in classes] if classes is not None else None
        )
        self.notes = notes

        # 运行级别的补充信息，例如参数量、设备、数据集名称
        self.meta: Dict[str, Any] = {}

        self.created_at = time.time()
        self.rows: List[Dict[str, float]] = []
        self.epochs: List[Dict[str, float]] = []

        self._epoch = 0
        self._step = 0
        self._y_true: List[int] = []
        self._y_pred: List[int] = []
        self._errors: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- 记录

    def info(self, **kwargs: Any) -> "Tracker":
        """补充运行信息，会显示在报告顶部（参数量、设备、数据集……）。"""
        self.meta.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    def log(self, epoch: Optional[int] = None, step: Optional[int] = None, **metrics: Any) -> "Tracker":
        """记录一个训练步的标量指标。

        非数值的键会被忽略，避免把字符串误当成曲线画出来。
        """
        if epoch is not None:
            self._epoch = int(epoch)
        if step is not None:
            self._step = int(step)

        row: Dict[str, float] = {"epoch": float(self._epoch), "step": float(self._step)}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue

        self.rows.append(row)
        self._step += 1
        return self

    def end_epoch(self, **extra: Any) -> "Tracker":
        """结束当前 epoch，把本段记录压成一条 epoch 均值。

        ``extra`` 用于补充只在验证阶段才有的指标，如 ``val_loss``、``val_acc``。
        """
        label = self.rows[-1]["epoch"] if self.rows else float(self._epoch)
        entry: Dict[str, float] = {"epoch": label}

        metric_keys = {k for row in self.rows for k in row if k not in _LABEL_KEYS}
        for key in metric_keys:
            values = [row[key] for row in self.rows if key in row]
            if values:
                entry[key] = float(np.mean(values))

        for key, value in extra.items():
            if value is None:
                continue
            try:
                entry[key] = float(value)
            except (TypeError, ValueError):
                continue

        self.epochs.append(entry)
        self.rows = []
        return self

    def add_predictions(self, y_true: Any, y_pred: Any, reset: bool = False) -> "Tracker":
        """累积验证集预测，用于混淆矩阵和分类指标。"""
        if reset:
            self._y_true.clear()
            self._y_pred.clear()

        true = np.asarray(y_true).ravel()
        pred = np.asarray(y_pred).ravel()
        if true.shape != pred.shape:
            raise ValueError(f"y_true 与 y_pred 长度不一致: {true.shape} vs {pred.shape}")

        self._y_true.extend(int(v) for v in true)
        self._y_pred.extend(int(v) for v in pred)
        return self

    def add_errors(
        self,
        images: Any,
        y_true: Any,
        y_pred: Any,
        scores: Optional[Any] = None,
        max_keep: int = 128,
    ) -> "Tracker":
        """留出错分样本用于拼图展示。

        ``images`` 支持 ``(N, H, W)`` 灰度或 ``(N, H, W, 3)`` 彩色。
        ``scores`` 一般传模型对预测类别的置信度，越低越优先展示。
        """
        images = np.asarray(images)
        true = np.asarray(y_true).ravel()
        pred = np.asarray(y_pred).ravel()
        if len(images) != len(true):
            raise ValueError("images 与标签数量不一致")

        conf = np.ones(len(true), dtype=float) if scores is None else np.asarray(scores, dtype=float).ravel()

        wrong = np.nonzero(true != pred)[0]
        if wrong.size == 0:
            return self

        # 置信度越低越值得看
        order = wrong[np.argsort(conf[wrong])]
        room = max(0, int(max_keep) - len(self._errors))
        for idx in order[:room]:
            self._errors.append(
                {
                    "image": images[idx].astype(np.float32),
                    "true": int(true[idx]),
                    "pred": int(pred[idx]),
                    "score": float(conf[idx]),
                }
            )
        return self

    # ---------------------------------------------------------------- 派生

    @property
    def elapsed(self) -> float:
        return time.time() - self.created_at

    @property
    def total_steps(self) -> int:
        return len(self.rows) + len(self.epochs)

    def label_of(self, value: int) -> str:
        if self.classes is not None and 0 <= value < len(self.classes):
            return self.classes[value]
        return str(value)

    def confusion_matrix(self) -> Optional[np.ndarray]:
        """基于已累积的预测计算混淆矩阵；没有预测时返回 None。"""
        if not self._y_true:
            return None

        true = np.asarray(self._y_true)
        pred = np.asarray(self._y_pred)

        if self.classes is not None:
            labels = list(range(len(self.classes)))
        else:
            labels = sorted(set(true.tolist()) | set(pred.tolist()))

        index = {label: i for i, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(true, pred):
            if t in index and p in index:
                cm[index[t], index[p]] += 1
        return cm

    def class_labels(self) -> List[str]:
        if self.classes is not None:
            return list(self.classes)
        return [str(v) for v in sorted(set(self._y_true) | set(self._y_pred))]

    # ---------------------------------------------------------------- 报告

    def report(self, out_dir: str = "reports", filename: Optional[str] = None) -> str:
        """生成 HTML 报告，返回文件路径。"""
        from pathlib import Path

        from .report import build_report

        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (filename or f"{_slug(self.name)}-report.html")
        path.write_text(build_report(self), encoding="utf-8")
        return str(path)


def _slug(text: str) -> str:
    keep = [c if c.isalnum() or c in "-_." else "-" for c in text.strip().lower()]
    slug = "".join(keep).strip("-")
    return slug or "run"
