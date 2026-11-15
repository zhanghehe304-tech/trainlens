"""trainlens —— 给训练过程拍一张照片的工具箱。

只做一件小事：把指标、验证集预测和错分样本记下来，最后产出一份能直接发出去的 HTML 报告。
"""

from .tracker import Tracker

__version__ = "0.1.0"
__all__ = ["Tracker", "__version__"]
