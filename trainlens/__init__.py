"""trainlens —— 给训练过程拍一张照片的工具箱。

只做一件小事：把指标、验证集预测和错分样本记下来，最后产出一份能直接发出去的 HTML 报告。
"""

from .tracker import Tracker

__version__ = "0.2.0"
__all__ = ["Tracker", "track", "__version__"]

# torch 只在真的要用时才导入，没装 PyTorch 的人照样能用核心功能
_LAZY = {"track": "torch_hooks"}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)
