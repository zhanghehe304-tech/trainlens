"""trainlens.torch_hooks —— PyTorch 适配器：一句 ``track(model)`` 开始记录梯度与激活。

用法::

    from trainlens import Tracker, track

    tracker = Tracker("resnet18", classes=classes)
    watch = track(model, tracker)

    for epoch in range(epochs):
        for step, (x, y) in enumerate(loader):
            ...
            loss.backward()
            watch.log(epoch=epoch, step=step, loss=loss.item(), acc=acc, lr=lr)
            optimizer.step()
            optimizer.zero_grad()
        watch.end_epoch(val_loss=val_loss, val_acc=val_acc)

    watch.report("reports/")   # 等价于 tracker.report()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

__all__ = ["TorchWatch", "track"]

# 权重层看分布，激活层看"死了多少"，两类都挂
_HOOK_TYPES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.SiLU,
)


def _label(name: str, module: nn.Module) -> str:
    """给层起一个有信息量的名字，报告里比裸编号好读。"""
    kind = type(module).__name__
    if isinstance(module, nn.Linear):
        return f"{name} · {kind}({module.in_features}→{module.out_features})"
    if isinstance(module, nn.Conv2d):
        return f"{name} · {kind}({module.in_channels}→{module.out_channels})"
    return f"{name} · {kind}"


class TorchWatch:
    """挂在 model 上的观测器：收集梯度范数与每层激活统计。"""

    def __init__(self, model: nn.Module, tracker, max_layers: int = 16, grad_norm: bool = True) -> None:
        self.model = model
        self.tracker = tracker
        self.max_layers = max_layers
        self.log_grad_norm = grad_norm

        self._stats: Dict[str, Dict[str, float]] = {}
        self._handles: list = []
        self._attach()

    # ---------------------------------------------------------------- 挂载

    def _attach(self) -> None:
        picked = 0
        for name, module in self.model.named_modules():
            if not name or not isinstance(module, _HOOK_TYPES):
                continue
            if picked >= self.max_layers:
                break
            picked += 1
            stats_key = _label(name, module)
            self._stats[stats_key] = {"n": 0.0, "sum": 0.0, "sum_sq": 0.0, "zeros": 0.0, "elems": 0.0}
            self._handles.append(module.register_forward_hook(self._make_hook(stats_key)))

    def _make_hook(self, name: str):
        def hook(module, inputs, output) -> None:
            if not torch.is_tensor(output) or output.numel() == 0:
                return
            with torch.no_grad():
                x = output.detach().float()
                stats = self._stats[name]
                stats["n"] += float(x.numel())
                stats["sum"] += float(x.sum())
                stats["sum_sq"] += float((x * x).sum())
                stats["zeros"] += float((x == 0).sum())
                stats["elems"] = float(x.numel())

        return hook

    def close(self) -> None:
        """摘掉所有 hook。不调用也不会出错，只是会一直挂着。"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "TorchWatch":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- 记录

    def grad_norm(self) -> float:
        """当前所有参数梯度的全局 L2 范数。要在 ``loss.backward()`` 之后、``zero_grad()`` 之前调。"""
        total = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            total += float(param.grad.detach().pow(2).sum())
        return total ** 0.5

    def log(self, epoch: Optional[int] = None, step: Optional[int] = None, **metrics: Any) -> "TorchWatch":
        """记录一步；默认顺手带上 ``grad_norm``。"""
        if self.log_grad_norm:
            metrics.setdefault("grad_norm", self.grad_norm())
        self.tracker.log(epoch=epoch, step=step, **metrics)
        return self

    def activations(self) -> Dict[str, Dict[str, float]]:
        """截至目前每层的 均值 / 标准差 / 置零比例（dead ratio）。"""
        summary: Dict[str, Dict[str, float]] = {}
        for name, stats in self._stats.items():
            n = stats["n"]
            if n <= 0:
                continue
            mean = stats["sum"] / n
            variance = max(stats["sum_sq"] / n - mean * mean, 0.0)
            summary[name] = {
                "mean": mean,
                "std": variance ** 0.5,
                "dead": stats["zeros"] / n,
                "elems": stats["elems"],
            }
        return summary

    def end_epoch(self, **extra: Any) -> "TorchWatch":
        """结束本轮：先把激活统计写进 tracker，再压 epoch 均值。"""
        self.tracker.add_activations(self.activations())
        self.tracker.end_epoch(**extra)
        for stats in self._stats.values():
            stats.update({"n": 0.0, "sum": 0.0, "sum_sq": 0.0, "zeros": 0.0})
        return self

    def report(self, out_dir: str = "reports", filename: Optional[str] = None) -> str:
        return self.tracker.report(out_dir, filename)


def track(model: nn.Module, tracker, max_layers: int = 16, grad_norm: bool = True) -> TorchWatch:
    """在 model 上挂一个观测器，并返回它。"""
    return TorchWatch(model, tracker, max_layers=max_layers, grad_norm=grad_norm)
