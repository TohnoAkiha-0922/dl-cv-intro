# 小型条件 U-Net：学习扩展与实验边界

## 这部分回答什么问题

在光学模型、采样和噪声已知的合成环境中，一个很小的网络能否学习图像复原？当离焦量换成训练时未出现的值，性能如何变化？先把这一条实验做完整，再考虑更复杂网络。

本项目是**已知离焦参数的条件复原**。网络收到模糊图和真实离焦量，维纳滤波收到由同一真实离焦量生成的 PSF。它们有相同的退化知识；不能把这个实验称为盲去模糊。

## 输入、网络和目标

- 第 1 通道：原始浮点观测强度，包含高斯噪声，不先截断。
- 第 2 通道：常数参数平面 `defocus_waves / 2.0`。2.0 是固定单位缩放，不是按每个数据集重新估计的统计量；超出训练范围也不截断。
- 目标：程序生成的 `[0,1]` 线性强度图，不是经过 sRGB 编码的相片。
- 两级下采样，默认通道数 8 / 16 / 32，最近邻上采样、跳接；卷积采用 circular padding，与本实验的周期成像约定相接。
- 网络预测加在观测上的残差。末层初始化为零，因此初始输出等于模糊观测，便于观察训练带来的变化。
- 训练损失是原始浮点输出与目标的 MSE，没有 sigmoid。统一在评价和显示前将**所有方法**截断到 `[0,1]`。

周期边界是教学仿真的简化。圆周卷积、圆周填充及有限图像可能造成与真实相机不同的边界行为。网络不显式执行维纳逆算子，也没有保证可恢复光学截止之外的信息。

## 数据分组和可复现性

`--train-count`、`--val-count`、`--test-count` 都指清晰底图的数量；每张底图与每个指定离焦量组合，因此样本数还要乘以离焦量的数量。训练、验证、测试分别使用 `scene_seed(split, index, base_seed)`，在产生模糊版本**之前**按底图来源划分。manifest 写出每个样本的 scene seed、noise seed 和离焦值，可检查是否重复。

噪声固定，便于重复比较；同一测试观测同时交给三种方法。默认训练和验证离焦量为 `0,0.4,0.8` waves。测试还包含未见的 `0.2,0.6,1.2`：前两者是在训练参数之间的插值，后者是超出训练范围的外推。测试底图对所有离焦量均未见。

训练仅依照验证集原始输出 MSE 选择 `best.pt`。不查看测试分数来选 epoch、调网络或调维纳参数。维纳正则系数默认固定 `0.01`，来自项目独立验证示例在候选 `1e-5,1e-4,0.001,0.01` 中的选择；它不是在神经网络测试结果上选择的。评估脚本本身不自动调参。更换数据或成像条件后，如需重新校准，应先使用验证集，并在测试前冻结。

模型参数、成像配置和训练设置都进入 checkpoint，使用纯 `state_dict` 和基本 Python 值保存，加载使用 `weights_only=True`。评估从 checkpoint 恢复像素间距、波长、F 数、图像尺寸、噪声和输入单位缩放；没有允许无声替换这些条件的命令行开关。

CPU、CUDA、不同 PyTorch 版本之间不保证逐位一致。已固定随机种子、DataLoader 顺序种子，关闭 cuDNN benchmark，并请求 cuDNN deterministic；实际运行环境写入配置。`--device auto` 会运行真实 CUDA 卷积及反向传播探测，发现硬件与当前 PyTorch 不兼容时明确警告并回退 CPU；显式 `--device cuda` 失败则退出并给出处理建议。

## 起步命令

在项目目录下，使用已安装兼容 PyTorch 的 Python 环境运行：

```powershell
python train.py --output outputs/smoke-network --epochs 2 --train-count 24 --val-count 8 --size 64 --batch-size 4 --device cpu
python evaluate.py --checkpoint outputs/smoke-network/best.pt --output outputs/smoke-evaluation --test-count 8 --device cpu
python -m unittest discover -s tests -p "test_learning.py" -v
```

需要使用显卡时将 `--device cpu` 换为 `--device auto`。正常学习可以从 `--size 128`、较多底图和训练轮次开始，先观察显存与验证损失，再决定是否增加规模。先查看 `python train.py --help` 和 `python evaluate.py --help`。

## 产物如何阅读

训练输出 `config.json`、`train-manifest.json`、`val-manifest.json`、`loss.csv`、验证选择的 `best.pt` 和 `training-summary.json`。

评估输出 `evaluation-config.json`、`test-manifest.json`、`per-sample.csv`、`aggregate.csv`、`summary.json`、`cross-defocus.png`。`representatives/` 中有每个离焦量第一张测试图的四图对比和原始 `.npz` 数组。CSV 的 PSNR 单位为 dB，SSIM 使用 11×11、sigma 1.5 的高斯窗、总体协方差，并排除 5 像素边缘。PSNR 按整张图计算，二者的边界口径不同，应如实记录。

先看训练和验证损失是否正常，再看传统方法和网络各自在什么图案、什么离焦量下失效。小样本训练后网络可能比维纳方法差，这同样是有效结果。不要仅因为网络更复杂就认定它更好。

## 结论的适用范围

这些是合成教学图像和理想化单色、非相干、空间不变成像；没有真实相机标定、彩色传感器、像素积分、光子噪声或空间变化 PSF。所附小规模训练用于检验数据、优化、重载、评价整条程序；本机已完成的 12 轮示例详见 `docs/validation.md`，不支持自然图像效果或科研创新的结论。后续应在老师指导下引入经标定的真实测量和独立真实数据验证。
