"""trainlens.report —— 把一次训练运行渲染成单文件 HTML（图片全部内联，可直接分享）。"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .viz import confusion_heatmap, error_grid, line_chart

__all__ = ["build_report"]

TRAIN_LOSS_KEYS = ("loss", "train_loss")
VAL_LOSS_KEYS = ("val_loss", "eval_loss", "test_loss")
TRAIN_ACC_KEYS = ("acc", "accuracy", "train_acc")
VAL_ACC_KEYS = ("val_acc", "val_accuracy", "eval_acc", "test_acc")
LR_KEYS = ("lr", "learning_rate")


def _pick(available: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in available:
            return name
    return None


def _frame(tracker) -> Tuple[List[float], List[Dict[str, float]], str]:
    """优先用 epoch 均值；没有 epoch 记录时退化成按 step 画。"""
    if tracker.epochs:
        x = [float(e.get("epoch", i)) for i, e in enumerate(tracker.epochs)]
        return x, tracker.epochs, "epoch"
    x = [float(r.get("step", i)) for i, r in enumerate(tracker.rows)]
    return x, tracker.rows, "step"


def _values(records: Sequence[Dict[str, float]], key: Optional[str]) -> List[float]:
    if not key:
        return []
    return [float(r[key]) for r in records if key in r]


def _class_metrics(cm: np.ndarray, labels: Sequence[str]) -> List[Dict[str, float | str]]:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)

    rows = []
    for i, name in enumerate(labels):
        rows.append(
            {
                "name": name,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
        )
    return rows


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def _section(title: str, body: str, extra_class: str = "") -> str:
    cls = f' class="{extra_class}"' if extra_class else ""
    return f'<section{cls}><h2>{html.escape(title)}</h2>{body}</section>'


def _chart_panel(title: str, b64: str) -> str:
    return (
        '<div class="panel">'
        f'<img alt="{html.escape(title)}" src="data:image/png;base64,{b64}">'
        "</div>"
    )


def build_report(tracker) -> str:
    """返回完整的 HTML 字符串。"""
    x, records, xlabel = _frame(tracker)
    available = {k for r in records for k in r} - {"epoch", "step"}

    # ---- 曲线
    charts: List[str] = []
    loss_series: Dict[str, List[float]] = {}
    train_loss = _pick(available, TRAIN_LOSS_KEYS)
    val_loss = _pick(available, VAL_LOSS_KEYS)
    if train_loss:
        loss_series["train loss"] = _values(records, train_loss)
    if val_loss:
        loss_series["val loss"] = _values(records, val_loss)

    acc_series: Dict[str, List[float]] = {}
    train_acc = _pick(available, TRAIN_ACC_KEYS)
    val_acc = _pick(available, VAL_ACC_KEYS)
    if train_acc:
        acc_series["train acc"] = _values(records, train_acc)
    if val_acc:
        acc_series["val acc"] = _values(records, val_acc)

    lr_key = _pick(available, LR_KEYS)

    if loss_series:
        charts.append(_chart_panel("Loss", line_chart(x, loss_series, "Loss", xlabel, "loss")))
    if acc_series:
        charts.append(_chart_panel("Accuracy", line_chart(x, acc_series, "Accuracy", xlabel, "acc")))
    if lr_key:
        lr_values = _values(records, lr_key)
        positive = all(v > 0 for v in lr_values)
        charts.append(
            _chart_panel(
                "Learning rate",
                line_chart(x, {"learning rate": lr_values}, "Learning rate", xlabel, "lr", logy=positive),
            )
        )

    if "grad_norm" in available:
        grad_values = _values(records, "grad_norm")
        positive = all(v > 0 for v in grad_values)
        charts.append(
            _chart_panel(
                "Gradient norm",
                line_chart(x, {"grad norm": grad_values}, "Gradient norm", xlabel, "L2 norm", logy=positive),
            )
        )

    # ---- 指标卡
    cards: List[str] = []
    best_val_acc = max(_values(records, val_acc)) if val_acc else None
    best_acc = max(_values(records, train_acc)) if train_acc else None
    final_loss = (_values(records, val_loss) or _values(records, train_loss) or [None])[-1]

    if best_val_acc is not None:
        cards.append(("best val acc", _fmt(best_val_acc)))
    elif best_acc is not None:
        cards.append(("best acc", _fmt(best_acc)))
    cards.append(("final loss", _fmt(final_loss)))
    epoch_count = len(tracker.epochs) if tracker.epochs else (int(records[-1].get("epoch", 0)) + 1 if records else 0)
    cards.append(("epochs", str(epoch_count)))
    cards.append(("elapsed", _human_time(tracker.elapsed)))
    if tracker.meta.get("params"):
        cards.append(("params", f"{int(tracker.meta['params']):,}"))

    # ---- 混淆矩阵与分类指标
    cm = tracker.confusion_matrix()
    labels = tracker.class_labels() if cm is not None else []
    class_panel = ""
    if cm is not None:
        try:
            heatmap = confusion_heatmap(cm, labels)
            table_rows = "".join(
                "<tr>"
                f"<td>{html.escape(str(row['name']))}</td>"
                f"<td>{_fmt(row['precision'])}</td>"
                f"<td>{_fmt(row['recall'])}</td>"
                f"<td>{_fmt(row['f1'])}</td>"
                f"<td>{row['support']}</td>"
                "</tr>"
                for row in _class_metrics(cm, labels)
            )
            accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
            class_panel = (
                '<div class="cols">'
                f'<div class="panel"><img alt="confusion matrix" src="data:image/png;base64,{heatmap}"></div>'
                '<div class="panel">'
                f'<p class="note">最后一轮验证集准确率 <strong>{_fmt(accuracy)}</strong>'
                f"（{int(cm.sum())} 个样本）</p>"
                "<table><thead><tr><th>class</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr>"
                f"</thead><tbody>{table_rows}</tbody></table>"
                "</div></div>"
            )
        except Exception as exc:  # 图渲染失败不该让整份报告打不开
            class_panel = f'<div class="panel"><p class="note">混淆矩阵渲染失败：{html.escape(str(exc))}</p></div>'

    # ---- 每层激活统计（torch 适配器产出）
    act_panel = ""
    activations = getattr(tracker, "activations", None)
    if activations:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(layer))}</td>"
            f"<td>{_fmt(values.get('mean'))}</td>"
            f"<td>{_fmt(values.get('std'))}</td>"
            f"<td>{_fmt(values.get('dead') * 100, 1)}%</td>"
            "</tr>"
            for layer, values in activations.items()
        )
        act_panel = (
            '<div class="panel">'
            '<p class="note">最后一轮、最后一次前向的统计。置零比例高，说明那层有大量神经元没被激活。</p>'
            "<table><thead><tr><th>layer</th><th>mean</th><th>std</th><th>zeros</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
            "</div>"
        )

    # ---- 错分样本
    errors_panel = ""
    errors = getattr(tracker, "_errors", [])
    if errors:
        shown = errors[:16]
        scores = [e["score"] for e in shown]
        grid = error_grid(
            [e["image"] for e in shown],
            [tracker.label_of(e["true"]) for e in shown],
            [tracker.label_of(e["pred"]) for e in shown],
            scores=scores,
        )
        errors_panel = _chart_panel("Errors", grid)

    # ---- 运行信息
    info_panel = ""
    if tracker.meta:
        items = "".join(
            f'<div class="kv"><span>{html.escape(str(k))}</span><b>{html.escape(str(v))}</b></div>'
            for k, v in tracker.meta.items()
        )
        info_panel = f'<div class="panel"><div class="kvlist">{items}</div></div>'

    created = datetime.fromtimestamp(tracker.created_at).strftime("%Y-%m-%d %H:%M")
    meta_bits = [f"generated {created}"]
    if x:
        meta_bits.append(f"{len(records)} {xlabel}s")

    notes_html = f'<p class="notes">{html.escape(tracker.notes)}</p>' if tracker.notes else ""

    cards_html = "".join(
        f'<div class="card"><div class="k">{html.escape(label)}</div><div class="v">{value}</div></div>'
        for label, value in cards
    )

    body_parts: List[str] = []
    if len(charts) > 1:
        body_parts.append(_section("Training curves", f'<div class="charts">{"".join(charts)}</div>'))
    elif charts:
        body_parts.append(_section("Training curves", charts[0]))
    else:
        body_parts.append(
            _section("Training curves", '<div class="panel"><p class="note">这次运行没有记录标量指标。</p></div>')
        )

    if class_panel:
        body_parts.append(_section("Validation", class_panel))
    if act_panel:
        body_parts.append(_section("Activations", act_panel))
    if errors_panel:
        body_parts.append(_section("Misclassified samples", errors_panel))
    if info_panel:
        body_parts.append(_section("Run info", info_panel))

    return TEMPLATE.format(
        title=html.escape(tracker.name),
        meta=html.escape(" · ".join(meta_bits)),
        notes=notes_html,
        cards=cards_html,
        body="".join(body_parts),
        generated=created,
    )


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · trainlens report</title>
<style>
:root {{
  --ink: #1a202c; --muted: #718096; --line: #e6e8eb;
  --bg: #f7f8fa; --accent: #2b6cb0;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif;
  line-height: 1.55; -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1040px; margin: 0 auto; padding: 36px 24px 72px; }}
header {{ background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 26px 28px; }}
.tag {{
  display: inline-block; font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); background: #ebf2fb; padding: 3px 9px; border-radius: 6px;
}}
h1 {{ margin: 14px 0 6px; font-size: 26px; font-weight: 650; letter-spacing: -.01em; }}
.meta {{ margin: 0; color: var(--muted); font-size: 13px; }}
.notes {{ margin: 10px 0 0; font-size: 13px; color: #4a5568; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 12px; margin-top: 24px; }}
.card {{ border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; background: #fcfcfd; }}
.card .k {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
.card .v {{ font-size: 20px; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }}
section {{ margin-top: 30px; }}
h2 {{ font-size: 15px; font-weight: 600; margin: 0 0 12px; padding-left: 10px; border-left: 3px solid var(--accent); }}
.panel {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 18px; }}
.panel img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }}
.cols {{ display: grid; grid-template-columns: 1.05fr 1fr; gap: 14px; align-items: start; }}
@media (max-width: 860px) {{ .cols {{ grid-template-columns: 1fr; }} }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 8px; border-bottom: 1px solid var(--line); }}
td {{ padding: 7px 8px; border-bottom: 1px solid #f1f2f4; }}
tbody tr:last-child td {{ border-bottom: none; }}
.note {{ margin: 0 0 12px; font-size: 13px; color: #4a5568; }}
.kvlist {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px 28px; }}
.kv {{ display: flex; justify-content: space-between; gap: 12px; font-size: 13px; border-bottom: 1px solid #f1f2f4; padding-bottom: 7px; }}
.kv span {{ color: var(--muted); }}
.kv b {{ font-weight: 600; text-align: right; word-break: break-word; }}
footer {{ margin-top: 34px; color: var(--muted); font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="tag">trainlens report</span>
    <h1>{title}</h1>
    <p class="meta">{meta}</p>
    {notes}
    <div class="cards">{cards}</div>
  </header>
  {body}
  <footer>Generated by trainlens · {generated}</footer>
</div>
</body>
</html>
"""
