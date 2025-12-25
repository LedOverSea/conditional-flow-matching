翻译自`runner/README.md`

在我们的`pytorch-lightning`实现中抽象出相关损失函数后，我们将所有`pytorch-lightning`代码移到了`runner`目录中。之前适用于lightning的所有命令现在都应从`runner`目录内运行。V0版本现已冻结在`V0`分支中。

使用默认配置训练模型

```bash
cd runner

# 在CPU上训练
python src/train.py trainer=cpu

# 在GPU上训练
python src/train.py trainer=gpu
```

使用从[configs/experiment/](configs/experiment/)中选择的实验配置训练模型

```bash
python src/train.py experiment=实验名称
```

你可以像这样从命令行覆盖任何参数

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

你也可以使用SLURM并行训练一系列模型，如`scripts/two-dim-cfm.sh`所示，该脚本训练了表2前三行使用的模型。

## 代码贡献

此仓库是从一个更大的私有代码库中提取出来的，因此丢失了包含论文其他作者工作的原始提交历史。

## 项目结构

新项目的目录结构如下：

```
│
│── runner                    <- Shell脚本
|   ├── data                   <- 项目数据
|   ├── logs                   <- 由hydra和lightning日志记录器生成的日志
|   ├── scripts                   <- Shell脚本
│   ├── configs                   <- Hydra配置文件
│   │   ├── callbacks                <- 回调函数配置
│   │   ├── debug                    <- 调试配置
│   │   ├── datamodule               <- 数据模块配置
│   │   ├── experiment               <- 实验配置
│   │   ├── extras                   <- 额外工具配置
│   │   ├── hparams_search           <- 超参数搜索配置
│   │   ├── hydra                    <- Hydra配置
│   │   ├── launcher                 <- Hydra启动器配置
│   │   ├── local                    <- 本地配置
│   │   ├── logger                   <- 日志记录器配置
│   │   ├── model                    <- 模型配置
│   │   ├── paths                    <- 项目路径配置
│   │   ├── trainer                  <- 训练器配置
│   │   │
│   │   ├── eval.yaml             <- 评估主配置
│   │   └── train.yaml            <- 训练主配置
│   ├── src                       <- 源代码
│   │   ├── datamodules              <- Lightning数据模块
│   │   ├── models                   <- Lightning模型
│   │   ├── utils                    <- 工具脚本
│   │   │
│   │   ├── eval.py                  <- 运行评估
│   │   └── train.py                 <- 运行训练
│   │
│   ├── tests                     <- 各类测试
│   └── README.md
```

## ⚡  你的超能力

<details>
<summary><b>从命令行覆盖任何配置参数</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **注意**：你也可以使用`+`号添加新参数。

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>在CPU、GPU、多GPU和TPU上训练</b></summary>

```bash
# 在CPU上训练
python train.py trainer=cpu

# 在1个GPU上训练
python train.py trainer=gpu

# 在TPU上训练
python train.py +trainer.tpu_cores=8

# 使用DDP（分布式数据并行）训练（4个GPU）
python train.py trainer=ddp trainer.devices=4

# 使用DDP（分布式数据并行）训练（8个GPU，2个节点）
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# 在CPU进程上模拟DDP
python train.py trainer=ddp_sim trainer.devices=2

# 在Mac上加速训练
python train.py trainer=mps
```

> **警告**：目前DDP模式存在问题，请阅读[此issue](https://github.com/ashleve/lightning-hydra-template/issues/393)以了解更多信息。

</details>

<details>
<summary><b>使用混合精度训练</b></summary>

```bash
# 使用PyTorch原生自动混合精度（AMP）训练
python train.py trainer=gpu +trainer.precision=16
```

</details>

<!-- deepspeed支持仍在测试中
<details>
<summary><b>使用Deepspeed在多GPU上优化大规模模型</b></summary>

```bash
python train.py +trainer.
```

</details>
 -->

<details>
<summary><b>使用PyTorch Lightning中可用的任何日志记录器训练模型，例如W&B或Tensorboard</b></summary>

```yaml
# 在`configs/logger/wandb`中设置项目和实体名称
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# 使用Weights&Biases训练模型（终端中应出现指向wandb仪表板的链接）
python train.py logger=wandb
```

> **注意**：Lightning为大多数流行的日志框架提供了便捷的集成。在[此处](#experiment-tracking)了解更多。

> **注意**：使用wandb需要你先[注册账户](https://www.wandb.com/)。之后只需完成如下配置。

> **注意**：点击[此处](https://wandb.ai/hobglob/template-dashboard/)查看使用此模板生成的示例wandb仪表板。

</details>

<details>
<summary><b>使用选定的实验配置训练模型</b></summary>

```bash
python train.py experiment=example
```

> **注意**：实验配置位于[configs/experiment/](configs/experiment/)。

</details>

<details>
<summary><b>附加一些回调函数来运行</b></summary>

```bash
python train.py callbacks=default
```

> **注意**：回调函数可用于模型检查点、提前停止等[诸多功能](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks)。

> **注意**：回调函数配置位于[configs/callbacks/](configs/callbacks/)。

</details>

<details>
<summary><b>使用PyTorch Lightning中可用的不同技巧</b></summary>

```yaml
# 可以启用梯度裁剪以避免梯度爆炸
python train.py +trainer.gradient_clip_val=0.5

# 在一个训练周期内运行4次验证循环
python train.py +trainer.val_check_interval=0.25

# 累积梯度
python train.py +trainer.accumulate_grad_batches=10

# 在12小时后终止训练
python train.py +trainer.max_time="00:12:00:00"
```

> **注意**：PyTorch Lightning提供了约[40多个有用的训练器标志](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags)。

</details>

<details>
<summary><b>轻松调试</b></summary>

```bash
# 在默认调试模式下运行1个周期
# 将日志目录更改为`logs/debugs/...`
# 将所有命令行日志记录器的级别设置为'DEBUG'
# 强制执行调试友好的配置
python train.py debug=default

# 仅使用1个批次运行1次训练、验证和测试循环
python train.py debug=fdr

# 打印执行时间分析
python train.py debug=profiler

# 尝试对1个批次过拟合
python train.py debug=overfit

# 如果张量中存在任何数值异常（如NaN或+/-inf），则引发异常
python train.py +trainer.detect_anomaly=true

# 记录模型的二阶梯度范数
python train.py +trainer.track_grad_norm=2

# 仅使用20%的数据
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **注意**：访问[configs/debug/](configs/debug/)查看不同的调试配置。

</details>

<details>
<summary><b>从检查点恢复训练</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **注意**：检查点可以是路径或URL。

> **注意**：当前加载检查点不会恢复日志记录器的实验，但未来的Lightning版本将支持此功能。

</details>

<details>
<summary><b>在测试数据集上评估检查点</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **注意**：检查点可以是路径或URL。

</details>

<details>
<summary><b>创建超参数扫描</b></summary>

```bash
# 这将按顺序运行6个实验，
# 每个实验使用不同的批次大小和学习率组合
python train.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **注意**：Hydra在作业启动时惰性地组合配置。如果你在启动作业/扫描后更改代码或配置，最终组合的配置可能会受到影响。

</details>

<details>
<summary><b>使用Optuna创建超参数扫描</b></summary>

```bash
# 这将运行`configs/hparams_search/mnist_optuna.yaml`中定义的超参数搜索，
# 覆盖选定的实验配置
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **注意**：使用[Optuna扫描器](https://hydra.cc/docs/next/plugins/optuna_sweeper)不需要你在代码中添加任何样板文件，所有内容都定义在[单个配置文件](configs/hparams_search/mnist_optuna.yaml)中。

> **警告**：Optuna扫描不具备容错性（如果一个作业崩溃，整个扫描都会崩溃）。

</details>

<details>
<summary><b>执行文件夹中的所有实验</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **注意**：Hydra为控制多运行行为提供了特殊语法。在[此处](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run)了解更多。上述命令执行[configs/experiment/](configs/experiment/)中的所有实验。

</details>

<details>
<summary><b>为多个不同的种子执行运行</b></summary>

```bash
python train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

> **注意**：`trainer.deterministic=True`使PyTorch更具确定性，但会影响性能。

</details>

<details>
<summary><b>在远程AWS集群上执行扫描</b></summary>

> **注意**：这应该可以通过使用[Hydra的Ray AWS启动器](https://hydra.cc/docs/next/plugins/ray_launcher)的简单配置来实现。此模板中尚未实现示例。

</details>

<!-- <details>
<summary><b>在SLURM集群上执行扫描</b></summary>

> 这应该可以通过[正确的lightning训练器标志](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster)或使用[Hydra的Submitit启动器](https://hydra.cc/docs/plugins/submitit_launcher)的简单配置来实现。此模板中尚未实现示例。

</details> -->

<details>
<summary><b>使用Hydra标签自动补全</b></summary>

> **注意**：Hydra允许你在shell中编写配置参数覆盖时，按下`tab`键自动补全。阅读[文档](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion)。

</details>

<details>
<summary><b>应用预提交钩子</b></summary>

```bash
pre-commit run -a
```

> **注意**：应用预提交钩子可以执行自动格式化代码和配置、进行代码分析或从jupyter笔记本中移除输出等操作。详见[#最佳实践](#最佳实践)。

</details>

<details>
<summary><b>运行测试</b></summary>

```bash
# 运行所有测试
pytest

# 运行特定文件的测试
pytest tests/test_train.py

# 运行除标记为慢速之外的所有测试
pytest -k "not slow"
```

</details>

<details>
<summary><b>使用标签</b></summary>

每个实验都应打上标签，以便在文件间或日志记录器UI中轻松筛选：

```bash
python train.py tags=["mnist","experiment_X"]
```

如果未提供标签，系统将要求你从命令行输入：

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - 正在强制执行标签！<cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - 配置中未提供标签。提示用户输入标签...
输入逗号分隔的标签列表（开发环境）：
```

如果多运行未提供标签，将引发错误：

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: 在启动多运行前指定标签！
```

> **注意**：当前hydra不支持从命令行追加列表 :(

</details>

<br>
[文件内容结束]