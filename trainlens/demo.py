"""trainlens.demo —— 零依赖的端到端示例：合成图形分类 + 纯 NumPy 的 MLP。

跑完之后 ``out/report.html`` 就能直接打开看，不需要装 PyTorch。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .tracker import Tracker

__all__ = ["make_shapes", "MLP", "run_demo"]

CLASSES = ["line", "bar", "diag"]


def _draw(images: np.ndarray, labels: np.ndarray, rng: np.random.Generator, size: int) -> None:
    for i, label in enumerate(labels):
        if label == 0:  # 横线
            row = rng.integers(3, size - 3)
            thickness = rng.integers(1, 3)
            images[i, row : row + thickness, 3 : size - 3] = 1.0
        elif label == 1:  # 竖线
            col = rng.integers(3, size - 3)
            thickness = rng.integers(1, 3)
            images[i, 3 : size - 3, col : col + thickness] = 1.0
        else:  # 斜线
            offset = rng.integers(-4, 5)
            coords = np.arange(3, size - 3)
            for k in coords:
                j = k + offset
                if 0 <= j < size:
                    images[i, k, j] = 1.0
                    if j + 1 < size:
                        images[i, k, j + 1] = 1.0


def make_shapes(
    n_per_class: int = 180,
    size: int = 14,
    noise: float = 0.38,
    ambiguous: float = 0.18,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成三类简单图形。``ambiguous`` 比例的样本会叠加第二种图形，制造真正的错分。"""
    rng = np.random.default_rng(seed)
    total = n_per_class * len(CLASSES)
    labels = np.repeat(np.arange(len(CLASSES)), n_per_class)
    rng.shuffle(labels)
    images = np.zeros((total, size, size), dtype=np.float32)

    _draw(images, labels, rng, size)

    # 少数样本叠上另一类的图形，训练后仍会有一部分判错
    for i in np.nonzero(rng.random(total) < ambiguous)[0]:
        other = (labels[i] + rng.integers(1, len(CLASSES))) % len(CLASSES)
        single = np.zeros((1, size, size), dtype=np.float32)
        _draw(single, np.array([other]), rng, size)
        images[i] = np.maximum(images[i], single[0] * 0.55)

    images += rng.normal(0, noise, images.shape).astype(np.float32)
    np.clip(images, 0, 1, out=images)
    return images, labels.astype(np.int64)


class MLP:
    """两层全连接 + ReLU，带动量 SGD。刻意写得朴素，方便在 README 里讲清楚。"""

    def __init__(self, n_in: int = 256, n_hidden: int = 64, n_out: int = 3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden)).astype(np.float64)
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out)).astype(np.float64)
        self.b2 = np.zeros(n_out)
        self._velocity = {k: np.zeros_like(v) for k, v in self.params().items()}

    def params(self) -> dict:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.params().values())

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0)
        logits = a1 @ self.w2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return z1, a1, probs

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, dict]:
        z1, a1, probs = self.forward(x)
        n = len(y)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grads = {
            "w2": a1.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        da1 = dlogits @ self.w2.T
        dz1 = da1 * (z1 > 0)
        grads["w1"] = x.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return loss, probs, grads

    def sgd_step(self, grads: dict, lr: float, momentum: float = 0.9) -> None:
        for name, value in self.params().items():
            v = self._velocity[name]
            v *= momentum
            v += grads[name]
            value -= lr * v

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[2].argmax(axis=1)


def run_demo(out_dir: str = "out", epochs: int = 8, seed: int = 0, size: int = 14, quiet: bool = False) -> str:
    """训练一遍并生成报告，返回报告路径。"""
    images, labels = make_shapes(size=size, seed=seed)
    split = int(len(images) * 0.75)
    x_train = images[:split].reshape(split, -1)
    y_train = labels[:split]
    x_val = images[split:].reshape(len(images) - split, -1)
    y_val = labels[split:]

    model = MLP(n_in=size * size, seed=seed)
    tracker = (
        Tracker("shapes-mlp", classes=CLASSES, notes="合成图形数据集 · 纯 NumPy 训练")
        .info(dataset=f"synthetic 3-class shapes {size}x{size}", train_size=int(len(y_train)),
              val_size=int(len(y_val)), params=int(model.n_params), optimizer="SGD + momentum 0.9")
    )

    lr0 = 0.12
    batch = 32
    rng = np.random.default_rng(seed + 1)
    steps_per_epoch = int(np.ceil(len(x_train) / batch))

    for epoch in range(epochs):
        # 余弦退火，报告里的 lr 曲线才有意义
        lr = lr0 * 0.5 * (1 + np.cos(np.pi * epoch / max(1, epochs)))
        order = rng.permutation(len(x_train))
        for step in range(steps_per_epoch):
            idx = order[step * batch : (step + 1) * batch]
            loss, probs, grads = model.loss_and_grads(x_train[idx], y_train[idx])
            model.sgd_step(grads, lr)
            acc = float((probs.argmax(axis=1) == y_train[idx]).mean())
            tracker.log(epoch=epoch, step=step, loss=loss, acc=acc, lr=lr)

        val_probs = model.forward(x_val)[2]
        val_pred = val_probs.argmax(axis=1)
        val_loss = float(-np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-12).mean())
        val_acc = float((val_pred == y_val).mean())

        tracker.end_epoch(val_loss=val_loss, val_acc=val_acc)
        tracker.add_predictions(y_val, val_pred, reset=True)

        if epoch == epochs - 1:
            confidence = val_probs.max(axis=1)
            tracker.add_errors(
                images[split:], y_val, val_pred, scores=confidence, max_keep=64
            )
            if not quiet:
                print(f"epoch {epoch + 1:>2}/{epochs}  loss {val_loss:.3f}  val_acc {val_acc:.3f}")

    path = tracker.report(out_dir)
    if not quiet:
        print(f"report: {Path(path).resolve()}")
    return path
