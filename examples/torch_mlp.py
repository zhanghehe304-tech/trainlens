"""把 trainlens 接进 PyTorch 训练循环的最小示例。

和 demo 用同一份合成数据，区别只在于这次模型和优化器来自 torch，
外加一句 ``track(model)`` 把梯度范数和每层激活统计也记下来：

    python examples/torch_mlp.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlens import Tracker, track
from trainlens.demo import CLASSES, make_shapes


def main(epochs: int = 12, batch_size: int = 32, lr: float = 0.05, out_dir: str = "out") -> str:
    images, labels = make_shapes(seed=0)
    x = torch.from_numpy(images).reshape(len(images), -1)
    y = torch.from_numpy(labels)

    split = int(len(x) * 0.75)
    train_loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=batch_size, shuffle=True)
    x_val, y_val = x[split:], y[split:]

    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    tracker = Tracker("torch-mlp", classes=CLASSES, notes="PyTorch 接入示例").info(
        dataset="synthetic shapes",
        params=int(sum(p.numel() for p in model.parameters())),
        device="cpu",
    )
    # 一句就挂上：之后每个 step 自动带 grad_norm，每个 epoch 自动写激活统计
    watch = track(model, tracker)

    for epoch in range(epochs):
        model.train()
        for step, (xb, yb) in enumerate(train_loader):
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()

            # 梯度还在，趁这时候记
            watch.log(
                epoch=epoch,
                step=step,
                loss=loss.item(),
                acc=(logits.argmax(dim=1) == yb).float().mean().item(),
                lr=optimizer.param_groups[0]["lr"],
            )
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val).item()
            probs = val_logits.softmax(dim=1)
            val_pred = probs.argmax(dim=1).numpy()
            confidence = probs.max(dim=1).values.numpy()

        val_acc = float((val_pred == y_val.numpy()).mean())
        watch.end_epoch(val_loss=val_loss, val_acc=val_acc)
        tracker.add_predictions(y_val.numpy(), val_pred, reset=True)
        tracker.add_errors(images[split:], y_val.numpy(), val_pred, scores=confidence)

        print(f"epoch {epoch + 1}/{epochs}  val_loss {val_loss:.3f}  val_acc {val_acc:.3f}")

    watch.close()
    return watch.report(out_dir)


if __name__ == "__main__":
    print(main())
