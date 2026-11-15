import tempfile
import unittest
from pathlib import Path

import numpy as np

from trainlens import Tracker


def make_tracker(**kwargs) -> Tracker:
    return Tracker("unit-test", classes=["a", "b", "c"], **kwargs)


class TrackingTest(unittest.TestCase):
    def test_end_epoch_averages_metrics(self):
        tracker = make_tracker()
        tracker.log(epoch=0, step=0, loss=1.0, acc=0.5)
        tracker.log(epoch=0, step=1, loss=3.0, acc=0.9)
        tracker.end_epoch(val_loss=2.2, val_acc=0.7)

        self.assertEqual(tracker.rows, [])
        self.assertEqual(len(tracker.epochs), 1)
        entry = tracker.epochs[0]
        self.assertAlmostEqual(entry["loss"], 2.0)
        self.assertAlmostEqual(entry["acc"], 0.7)
        self.assertAlmostEqual(entry["val_loss"], 2.2)
        self.assertEqual(entry["epoch"], 0.0)

    def test_non_numeric_metrics_are_ignored(self):
        tracker = make_tracker()
        tracker.log(epoch=0, loss=1.0, note="不是数值")
        self.assertIn("loss", tracker.rows[0])
        self.assertNotIn("note", tracker.rows[0])

    def test_confusion_matrix(self):
        tracker = make_tracker()
        tracker.add_predictions([0, 0, 1, 1], [0, 1, 0, 1])
        cm = tracker.confusion_matrix()
        self.assertIsNotNone(cm)
        np.testing.assert_array_equal(cm, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]))

    def test_confusion_matrix_without_classes(self):
        tracker = Tracker("no-classes")
        tracker.add_predictions([0, 1, 2], [0, 2, 1])
        cm = tracker.confusion_matrix()
        self.assertEqual(tracker.class_labels(), ["0", "1", "2"])
        np.testing.assert_array_equal(cm, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

    def test_add_predictions_length_mismatch(self):
        tracker = make_tracker()
        with self.assertRaises(ValueError):
            tracker.add_predictions([0, 1], [0])

    def test_errors_keep_only_misclassified_by_confidence(self):
        tracker = make_tracker()
        images = np.zeros((4, 8, 8), dtype=np.float32)
        true = [0, 1, 2, 0]
        pred = [0, 0, 1, 1]
        scores = [0.9, 0.55, 0.31, 0.72]
        tracker.add_errors(images, true, pred, scores=scores)

        # 三个错分样本里，置信度低的排前面
        self.assertEqual([e["pred"] for e in tracker._errors], [1, 0, 1])
        self.assertEqual([e["score"] for e in tracker._errors], [0.31, 0.55, 0.72])

    def test_errors_respect_max_keep(self):
        tracker = make_tracker()
        images = np.zeros((5, 6, 6), dtype=np.float32)
        tracker.add_errors(images, [0] * 5, [1] * 5, max_keep=2)
        self.assertEqual(len(tracker._errors), 2)


class ReportTest(unittest.TestCase):
    def test_report_contains_core_sections(self):
        tracker = Tracker("demo-run", classes=["a", "b"], notes="单测用")
        for epoch in range(3):
            tracker.log(epoch=epoch, step=0, loss=1.0 / (epoch + 1), acc=0.5 + epoch * 0.1, lr=0.01)
            tracker.end_epoch(val_loss=0.8, val_acc=0.6)
        tracker.add_predictions([0, 0, 1, 1], [0, 1, 0, 1])
        tracker.add_errors(np.zeros((4, 8, 8), dtype=np.float32), [0, 0, 1, 1], [0, 1, 0, 1])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(tracker.report(directory))
            self.assertTrue(path.exists())
            html = path.read_text(encoding="utf-8")

        self.assertIn("demo-run", html)
        self.assertIn("trainlens report", html)
        self.assertIn("Validation", html)
        self.assertIn("最后一轮验证集准确率", html)
        self.assertNotIn("渲染失败", html)
        self.assertIn("Misclassified samples", html)
        self.assertGreaterEqual(html.count("data:image/png;base64,"), 3)

    def test_report_without_metrics_still_renders(self):
        tracker = Tracker("empty")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(tracker.report(directory))
            html = path.read_text(encoding="utf-8")
        self.assertIn("empty", html)
        self.assertIn("没有记录标量指标", html)


class DemoTest(unittest.TestCase):
    def test_demo_runs_end_to_end(self):
        from trainlens.demo import run_demo

        with tempfile.TemporaryDirectory() as directory:
            path = Path(run_demo(out_dir=directory, epochs=2, quiet=True))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
