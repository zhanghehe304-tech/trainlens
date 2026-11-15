"""PyTorch 适配器的单测。没装 torch 的机器会自动跳过。"""

import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - 取决于环境
    import torch
    from torch import nn

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

from trainlens import Tracker


def make_model() -> "nn.Module":
    return nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 3))


def _track(model, tracker, **kwargs):
    """延迟导入，没装 torch 的机器连 import 都不会失败。"""
    from trainlens import track

    return track(model, tracker, **kwargs)


@unittest.skipUnless(HAS_TORCH, "未安装 torch")
class TorchHookTest(unittest.TestCase):
    def _train_step(self, model, tracker, watch):
        x = torch.randn(4, 8)
        y = torch.tensor([0, 1, 2, 0])
        criterion = nn.CrossEntropyLoss()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        watch.log(epoch=0, step=0, loss=loss.item(), acc=0.5)
        return loss

    def test_track_records_grad_norm(self):
        tracker = Tracker("torch-test", classes=["a", "b", "c"])
        model = make_model()
        watch = _track(model, tracker)

        self._train_step(model, tracker, watch)

        self.assertEqual(len(tracker.rows), 1)
        self.assertGreater(tracker.rows[0]["grad_norm"], 0.0)
        # 逐层梯度：两个带权重的 Linear 各一条，ReLU 没参数不参与
        layer_keys = [k for k in tracker.rows[0] if k.startswith("grad:")]
        self.assertEqual(len(layer_keys), 2)
        watch.close()

    def test_activations_have_layer_stats(self):
        tracker = Tracker("torch-test", classes=["a", "b", "c"])
        model = make_model()
        watch = _track(model, tracker)
        self._train_step(model, tracker, watch)

        watch.end_epoch(val_loss=0.5, val_acc=0.5)

        self.assertEqual(len(tracker.epochs), 1)
        self.assertAlmostEqual(tracker.epochs[0]["val_loss"], 0.5)
        self.assertEqual(len(tracker.activations), 3)  # 两个 Linear + 一个 ReLU
        for stats in tracker.activations.values():
            self.assertIn("mean", stats)
            self.assertGreaterEqual(stats["dead"], 0.0)
            self.assertLessEqual(stats["dead"], 1.0)
        # 名字带上了形状，报告里能直接看出是哪一层
        self.assertTrue(any("Linear(8→6)" in key for key in tracker.activations))

    def test_report_contains_gradient_and_activation_sections(self):
        tracker = Tracker("torch-test", classes=["a", "b", "c"])
        model = make_model()
        watch = _track(model, tracker)
        for step in range(2):
            self._train_step(model, tracker, watch)
        watch.end_epoch(val_loss=0.4, val_acc=0.6)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(watch.report(directory))
            html = path.read_text(encoding="utf-8")
        watch.close()

        self.assertIn("Activations", html)
        self.assertIn("Gradient norm", html)  # 图片 alt 文本
        self.assertIn("by layer", html)

    def test_conv_label_has_kernel_and_stride(self):
        # 卷积层的层名要能看出感受野，不然一堆 Conv2d(3→32) 分不清
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2), nn.ReLU())
        tracker = Tracker("torch-test", top_layers=3)
        watch = _track(model, tracker)
        labels = list(watch._stats)
        self.assertTrue(any("k3x3 s2x2" in name for name in labels))
        self.assertEqual(tracker.top_layers, 3)  # 逐层梯度图最多画几层
        watch.close()

    def test_close_is_idempotent(self):
        model = make_model()
        tracker = Tracker("torch-test")
        watch = _track(model, tracker, max_layers=1)
        self.assertEqual(len(watch._handles), 1)
        watch.close()
        watch.close()
        self.assertEqual(len(watch._handles), 0)


if __name__ == "__main__":
    unittest.main()
