# trainlens

[![CI](https://github.com/zhanghehe304-tech/trainlens/actions/workflows/ci.yml/badge.svg)](https://github.com/zhanghehe304-tech/trainlens/actions/workflows/ci.yml)

> 训练跑完，把过程拍成一张照片。

记录标量指标、验证集预测和错分样本，最后产出一份**单文件 HTML 报告**：训练曲线、混淆矩阵、逐类指标、错分样本网格全在里面。不用起 TensorBoard，也不用写一堆 matplotlib 样板代码。PyTorch 用户再加一句 `track(model)`，梯度范数和每层激活统计也一并进报告。

![report preview](https://aka.doubaocdn.com/s/WBOoxjNwpK)

## 为什么做这个

复盘一次训练，通常得在 notebook 里手写四五张图：loss 曲线、混淆矩阵、每类指标、错分样本……每换一个项目就抄一遍，而且彼此样式还不一样。

TensorBoard 适合边训边看，但它要起服务、占端口，报告也没法直接发给别人。这个工具解决的是另一半需求：**训练结束之后，留一份能截图、能发群、能贴进周报的东西**。

## 快速开始

```bash
pip install -e .
trainlens demo
```

十秒左右会在 `out/shapes-mlp-report.html` 生成一份完整报告，浏览器直接打开即可。

demo 用的是程序合成的三分类图形数据（横线 / 竖线 / 斜线）加一个纯 NumPy 写的两层 MLP，**不需要装 PyTorch**，克隆下来就能跑通全流程，也顺便验证了这套 API 不绑定任何框架。

## 接到自己的训练循环

```python
from trainlens import Tracker

tracker = Tracker("resnet18-cifar10", classes=class_names, notes="lr 0.1 + cosine")
tracker.info(
    dataset="CIFAR-10",
    params=sum(p.numel() for p in model.parameters()),
    device=str(device),
)

for epoch in range(epochs):
    for step, (x, y) in enumerate(train_loader):
        ...
        loss, acc = train_step(x, y)
        tracker.log(epoch=epoch, step=step, loss=loss, acc=acc, lr=scheduler.get_last_lr()[0])

    val_loss, val_acc, y_true, y_pred, conf = evaluate(model, val_loader)
    tracker.end_epoch(val_loss=val_loss, val_acc=val_acc)
    tracker.add_predictions(y_true, y_pred, reset=True)
    tracker.add_errors(x_val, y_true, y_pred, scores=conf)

path = tracker.report("reports/")
```

要点只有三个：训练步里 `log()`，轮末 `end_epoch()` 把验证指标一起塞进去，验证完把预测和错分样本交给 tracker。`reset=True` 很关键——混淆矩阵只保留最后一轮，不然会越滚越糊。

## 梯度与激活（PyTorch）

PyTorch 用户还能再省一步：`track(model, tracker)` 会挂上 forward hook，之后每个 step 顺手记下梯度范数，每轮结束自动写每层激活统计。

```python
from trainlens import Tracker, track

tracker = Tracker("resnet18", classes=classes)
watch = track(model, tracker)          # 一句开始观测

for epoch in range(epochs):
    for step, (x, y) in enumerate(loader):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()

        # 梯度还在，趁这时候记；grad_norm 会自动带上
        watch.log(epoch=epoch, step=step, loss=loss.item(), acc=acc, lr=lr)
        optimizer.step()

    watch.end_epoch(val_loss=val_loss, val_acc=val_acc)
watch.close()
```

报告里会多出一张梯度范数曲线（对数坐标）和一张逐层统计表：均值、标准差、置零比例。置零比例就是「这层有多少神经元彻底不响」，ReLU 层健康时通常不该高得离谱。

只挂 `Linear / Conv / ReLU / LeakyReLU / GELU / SiLU`，最多 16 层，够覆盖常见模型，也不会把大模型拖慢。核心包不 import torch，没装 PyTorch 完全不影响使用。

## API

| 方法 | 说明 |
| --- | --- |
| `Tracker(name, classes=None, notes="")` | 创建一次运行；`classes` 用来给类别起名 |
| `.info(**kv)` | 补充运行信息（数据集、参数量、设备），显示在报告里 |
| `.log(epoch=None, step=None, **metrics)` | 记录一个训练步；非数值的键会被忽略 |
| `.end_epoch(**extra)` | 结束本轮，压成均值；`extra` 用来传 `val_loss`、`val_acc` |
| `.add_predictions(y_true, y_pred, reset=False)` | 累积验证集预测，用于混淆矩阵 |
| `.add_errors(images, y_true, y_pred, scores=None, max_keep=128)` | 留错分样本，置信度越低越靠前 |
| `.report(out_dir="reports")` | 生成报告，返回文件路径 |
| `track(model, tracker)` | PyTorch：挂 hook，返回 `watch`（需要额外安装 torch） |
| `watch.log(...)` | 同 `Tracker.log`，额外自动记录 `grad_norm` |
| `watch.end_epoch(**extra)` | 先写入每层激活统计，再压 epoch 均值 |
| `watch.close()` | 摘掉 hook（也可以 `with track(...) as watch`） |

支持的图像形状：`(N, H, W)` 灰度、`(N, H, W, 3)` 彩色。

## 报告里有什么

顶部是一排指标卡：最佳验证准确率、最终 loss、轮数、耗时、参数量。

接着三张曲线：loss、准确率、学习率——学习率自动转对数坐标，步数超过 50 时自动叠一条平滑线，原始点保留成淡色底纹。用了 `track()` 的话还会多一张梯度范数曲线，同样走对数坐标，训练后期异常反弹一眼就能看出来。

然后是验证部分：混淆矩阵热力图，配一张逐类 precision / recall / f1 / support 的表。接 PyTorch 时后面再跟一张激活统计表，逐层给出均值、标准差和置零比例。

最后是错分样本网格，按模型对预测类别的置信度从低到高排——排在前面那些，往往就是「它凭什么错」的答案。

所有图片都是 base64 内联的，整份报告就是一个 HTML 文件，可以直接发出去。

## 几个设计取舍

**不绑定框架。** 核心只依赖 numpy，matplotlib 只在渲染报告时才会导入。PyTorch、JAX、纯 NumPy 都能接，代价是它不认识你的 `DataLoader`，得自己把预测喂进来。

**不做 TensorBoard 的替代品。** 边训边看还是用 TensorBoard / wandb。这里做的是训练结束后的存档与分享，两者不冲突。

**产出必须是单文件。** 报告要能离开你的机器。带个 `assets/` 目录就没法直接发到聊天窗口里了。

## 目录结构

```
trainlens/
├── trainlens/
│   ├── tracker.py    # 记录指标、预测、错分样本
│   ├── report.py     # 组装 HTML
│   ├── viz.py        # 曲线、热力图、样本网格
│   ├── torch_hooks.py # track(model)：梯度范数 + 激活统计
│   ├── demo.py       # 合成数据 + NumPy MLP
│   └── cli.py        # trainlens demo
├── examples/         # 各种框架的接入示例
├── tests/
└── .github/workflows/ci.yml
```

## 开发

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

CI 在 Python 3.10 / 3.12 / 3.13 上跑单测，外加一次 demo 冒烟。

## Roadmap

- 报告同时导出 CSV / JSON，方便跨实验对比画图
- 逐层梯度范数，而不只是全局范数
- 深色主题与自定义模板
- HuggingFace Trainer、Lightning 的 callback

## License

MIT
