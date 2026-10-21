# 人工智能作业：简化 YOLOv1 手写数字检测

学生：吉阳（250980418）

这是 YOLOv1 手写数字检测课程作业。目录包含可运行代码、教师原始模板、训练好的权重、MNIST 数据、PDF 报告，以及作业要求的可视化结果。

## 目录内容

- `create_dataset.py`：从 MNIST 合成 64×64 手写数字检测数据。
- `model.py`：构建输出 `(N, 7, 7, 20)` 的简化 YOLOv1。
- `loss.py`：IoU、负责框匹配、坐标/置信度/类别损失。
- `nms.py`：按类别执行非极大值抑制。
- `train.py`：训练、验证、测试评估和结果导出。
- `utils.py`：检测可视化、网络结构图和损失曲线。
- `predict.py`：加载已训练权重进行演示。
- `test_core.py`：12 项核心逻辑测试。
- `build_report.py`：根据实验记录生成 PDF 报告。
- `outputs/yolov1_digit.pth`：最佳模型权重。
- `outputs/network.png`：网络结构图。
- `outputs/loss_curves.png`：总损失、坐标损失、置信度损失和验证损失曲线。
- `outputs/dataset_examples.png`：合成数据可视化。
- `outputs/test_01.png` 至 `test_03.png`：至少三张测试结果图，绿色为真实框、红色为预测框。
- `outputs/failure_case.png`：失败案例。
- `output/pdf/report_250980418.pdf`：包含理解题、损失说明、测试分析和失败案例的报告。
- `teacher_originals/`：教师提供的原始代码，用于对照。
- `data/MNIST/raw/`：MNIST 原始数据，可离线使用。

## 运行

安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

加载权重并预测：

```powershell
python predict.py --device cpu
```

运行自动测试：

```powershell
python -m unittest -v test_core
```

使用支持 CUDA 的 PyTorch 环境可以把 `--device cpu` 改为 `--device cuda`。如需重新训练：

```powershell
python train.py --epochs 40 --batch-size 64 --train-samples 16000 --refresh-every 1
```

MNIST 数据集使用 torchvision 的标准原始文件布局，因此无需网络下载。

## 最终测试结果

1500 张独立测试图片上，TP=3350、FP=531、FN=421；Precision=86.32%、Recall=88.84%、F1=87.56%，正确匹配框平均 IoU=0.731。置信度阈值依据验证集选择为 0.4，NMS IoU 阈值为 0.5。评估要求类别正确且 IoU≥0.5，每个真实目标最多匹配一次。
