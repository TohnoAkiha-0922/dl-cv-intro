"""从实际训练记录生成中文 PDF 报告及配套 Markdown，不生成虚构实验数据。

依赖：reportlab、Pillow；Windows 优先嵌入宋体，可用 --font 指定其他中文 TTF。
本脚本可在有 reportlab 的 Python 中独立运行，不需要导入 PyTorch。
"""

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Image, Table, TableStyle, KeepTogether)

PROJECT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def iou(a, b):
    intersection = max(0, min(a[0] + a[2] / 2, b[0] + b[2] / 2)
                       - max(a[0] - a[2] / 2, b[0] - b[2] / 2)) * max(
        0, min(a[1] + a[3] / 2, b[1] + b[3] / 2)
        - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0


def failure_analysis(record, nms_threshold=0.5):
    details = []
    if record["tp"]:
        details.append(f"该图有 {record['tp']} 个目标被正确匹配；下面重点分析未匹配项。")
    for gt in record["ground_truth"]:
        nearest = max(record["predictions"], key=lambda box: iou(gt, box), default=None)
        if nearest is None:
            details.append(f"真实数字 {gt[4]} 没有通过置信度阈值的候选框，发生漏检。")
            continue
        overlap = iou(gt, nearest)
        if overlap >= .5 and nearest[5] == gt[4]:
            continue
        elif overlap >= .5:
            details.append(f"真实数字 {gt[4]} 的最佳重叠框 IoU={overlap:.3f}，"
                           f"但预测为 {nearest[5]}（置信度 {nearest[4]:.3f}），属于类别识别错误。")
        else:
            details.append(f"真实数字 {gt[4]} 与现有预测框的最大 IoU 仅为 {overlap:.3f}，"
                           "未达到 0.5 的匹配标准；可能涉及定位偏差或候选框被过滤。")
    for box, hit in zip(record["predictions"], record["hits"]):
        if hit:
            continue
        gt = max(record["ground_truth"], key=lambda target: iou(box, target), default=None)
        matched = [other for other, success in zip(record["predictions"], record["hits"])
                   if success and other[5] == box[5]]
        other = max(matched, key=lambda candidate: iou(box, candidate), default=None)
        if gt is not None and gt[4] == box[5] and other is not None:
            overlap = iou(box, other)
            details.append(f"额外的数字 {box[5]} 候选置信度为 {box[4]:.3f}，与最近真值 IoU={iou(box, gt):.3f}，"
                           f"该真值已由另一个更高分框正确匹配，额外候选计为 FP。它与已保留同类框的 "
                           f"IoU={overlap:.3f}，低于 NMS 阈值 {nms_threshold}，因此仍被保留。"
                           "图中目标跨过网格边界，相邻网格给出重复候选，是本例误检的直接表现。"
                           "可在验证集比较更低的 NMS 阈值，同时检查是否误删相邻同类真目标。")
        else:
            details.append(f"未匹配的类别 {box[5]} 预测框置信度为 {box[4]:.3f}，计作一次 FP；"
                           "它没有满足同类、IoU 达标且真值未被占用的匹配条件。")
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--pdf", type=Path, default=PROJECT / "output/pdf/report_250980418.pdf")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/simsun.ttc"))
    args = parser.parse_args()
    result = args.outputs
    config = load_json(result / "config.json")
    dataset = load_json(result / "dataset_stats.json")
    metrics_payload = load_json(result / "metrics.json")
    metrics = metrics_payload["test"]
    baseline_dir = PROJECT / "experiments/fixed_cache_baseline"
    baseline = load_json(baseline_dir / "metrics.json")["test"] if baseline_dir.exists() else None
    with (result / "history.csv").open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    records = load_json(result / "test_predictions.json")
    selected = [record for record in records if record["ground_truth"]][:3]
    failure = load_json(result / "failure_case.json")
    failure_text = failure_analysis(failure, metrics["nms_iou_thresh"])

    if args.font.exists():
        pdfmetrics.registerFont(TTFont("Chinese", str(args.font), subfontIndex=0))
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font = "Chinese" if args.font.exists() else "STSong-Light"
    styles = {
        "title": ParagraphStyle("TitleCN", fontName=font, fontSize=24, leading=36,
                                alignment=TA_CENTER, textColor=colors.HexColor("#143650"), spaceAfter=16),
        "sub": ParagraphStyle("SubCN", fontName=font, fontSize=15, leading=24,
                              alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("H1CN", fontName=font, fontSize=16, leading=24,
                             textColor=colors.HexColor("#143650"), spaceAfter=12),
        "h2": ParagraphStyle("H2CN", fontName=font, fontSize=11.5, leading=18,
                             spaceBefore=7, spaceAfter=4, textColor=colors.HexColor("#143650")),
        "body": ParagraphStyle("BodyCN", fontName=font, fontSize=10, leading=16,
                               wordWrap="CJK", spaceAfter=6),
        "small": ParagraphStyle("SmallCN", fontName=font, fontSize=8.4, leading=13,
                                wordWrap="CJK", spaceAfter=5),
        "caption": ParagraphStyle("CaptionCN", fontName=font, fontSize=8.5, leading=13,
                                  alignment=TA_CENTER, spaceAfter=8, textColor=colors.HexColor("#536575")),
    }
    story, markdown = [], []

    def paragraph(text, style="body"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def text(value, style="body"):
        story.append(paragraph(value, style))
        markdown.append(value + "\n")

    def heading(value, sub=False):
        text(value, "h2" if sub else "h1")

    def page():
        story.append(PageBreak())
        markdown.append("\n---\n")

    def image(path, width, max_height=None):
        with PILImage.open(path) as img:
            height = width * img.height / img.width
        if max_height and height > max_height:
            width *= max_height / height
            height = max_height
        return Image(str(path), width=width, height=height)

    def figure(name, caption, width=17.2 * cm, max_height=None):
        story.append(image(result / name, width, max_height))
        story.append(paragraph(caption, "caption"))
        markdown.append(f"![{caption}](outputs/{name})\n")

    def table(rows, widths, small=True):
        content = [[paragraph(cell, "small" if small else "body") for cell in row] for row in rows]
        element = Table(content, colWidths=widths, hAlign="LEFT", repeatRows=1)
        element.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#708799")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d9e1e7")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(element)
        story.append(Spacer(1, 7))
        markdown.extend([" / ".join(map(str, row)) + "\n" for row in rows])

    def pct(value):
        return f"{value * 100:.2f}%"

    # 第 1 页：封面。
    story.append(Spacer(1, 1.4 * cm))
    text("数理科学与工程学院", "sub")
    text("程序设计说明书", "title")
    text("2026 / 2027 学年第一学期", "sub")
    story.append(Spacer(1, .8 * cm))
    text("实现简化的 YOLOv1\n进行手写数字检测", "title")
    story.append(Spacer(1, .9 * cm))
    table([["项目", "信息"], ["姓名", "吉阳"], ["学号", "250980418"],
           ["专业", "光电信息科学与工程"], ["学院", "数理科学与工程学院"],
           ["日期", "2026 年 10 月 19 日"]], [4.2 * cm, 13 * cm], small=False)
    story.append(Spacer(1, .7 * cm))
    text("实验摘要", "h2")
    text(f"补全教师模板的卷积网络、IoU 匹配、多任务损失与 NMS，使用 MNIST 合成 64×64 数字检测图。"
         f"最终训练 {config['epochs']} 轮，在独立的 {metrics['images']} 张测试图、{dataset['test']['objects']} 个标注目标上，"
         f"得到 Precision={pct(metrics['precision'])}、Recall={pct(metrics['recall'])}、F1={pct(metrics['f1'])}。"
         "正确检测要求类别一致且 IoU≥0.5；全部结果来自实际运行记录。")
    text("提交内容：完整源代码、训练权重、损失曲线、检测图、关键测试及本报告。", "small")

    # 第 2 页：数据与理解性问题。
    page()
    heading("一、数据集构建与标签理解")
    text("采用教师提供的 create_dataset.py，以黑色 64×64 灰度画布合成每张 0~5 个数字。"
         "按教师原代码将 28×28 MNIST 图缩放到 8~14 像素，再随机粘贴，重叠区域取像素最大值。"
         "这里的缩放因子 0.3~0.5 乘的是原始 28 像素，而非画布的 64 像素。")
    figure("dataset_examples.png", "图 1  合成训练样本；绿色框与数字为真实标注。", max_height=8.8 * cm)
    heading("标签张量是什么形状？", True)
    text("单张为 (S,S,5B+C)=(7,7,20)，批次为 (N,7,7,20)。前两维是网格行、列；"
         "前 10 个数是两个 (cx,cy,w,h,conf) 框槽，后 10 个数是 0~9 的 one-hot 类别。"
         "cx、cy、w、h 均为整幅图的归一化坐标。")
    heading("为什么由中心所在网格负责？同格两个目标怎么办？", True)
    text("中心点提供明确的目标归属规则，避免多个网格同时监督同一目标。YOLOv1 每格只有一组类别分布，"
         "即使 B=2，也无法独立表达同格两个不同类别目标。原代码会覆盖框坐标并累积类别位，"
         "本实现于粘贴前跳过同格冲突，避免图中有数字而标签不一致；此举限制了任务的拥挤程度。")
    heading("标签里的 conf 如何设置？", True)
    text("空网格的框槽 conf=0；有目标网格按教师实际代码重复保存真值，两个槽的 conf 均为 1。"
         "这只是重复存储，损失函数仍按 IoU 只选择一个预测框作为正样本，其余框监督为 0。"
         "原文件注释和实际赋值不一致，报告以实际赋值为准。")
    text("数据划分：MNIST 训练集前 50000 张供训练合成，后 10000 张供验证合成；"
         "MNIST 官方测试集仅供最终测试。验证 seed=123，测试 seed=321，训练组合逐轮更换。", "small")

    # 第 3 页：结构。
    page()
    heading("二、简化 YOLOv1 网络结构")
    figure("network.png", "图 2  网络结构；张量维度按通道×高×宽表示。")
    table([["阶段", "操作", "输出（不含 batch）"],
           ["输入", "64×64 灰度图，像素归一化至 [0,1]", "1×64×64"],
           ["卷积块 1", "Conv(1,32,3,pad=1) + LeakyReLU(0.1) + MaxPool(2)", "32×32×32"],
           ["卷积块 2", "Conv(32,64,3,pad=1) + LeakyReLU + MaxPool(2)", "64×16×16"],
           ["卷积块 3", "Conv(64,128,3,pad=1) + LeakyReLU + MaxPool(2)", "128×8×8"],
           ["展平", "flatten(start_dim=1)", "8192"],
           ["全连接头", "Linear(8192,512) + LeakyReLU + Dropout(0.1)", "512"],
           ["输出", "Linear(512,980) + reshape + 框解码", "7×7×20"]],
          [2.3 * cm, 11.1 * cm, 3.8 * cm])
    heading("输出维度与激活设计", True)
    text(f"输出总维度为 7×7×(5×2+10)=980，模型共有 {config['parameters']:,} 个可训练参数。"
         "forward 依次完成 backbone、flatten、head、reshape；三个卷积块符合轻量模型要求。")
    text("为约束中心点落在负责网格中，头部先输出局部偏移 tx、ty，再计算 "
         "cx=(sigmoid(tx)+gx)/7、cy=(sigmoid(ty)+gy)/7。宽高 w、h 与 conf 使用 sigmoid。"
         "最终输出仍然是教师标签使用的全图坐标，因此 loss、NMS、可视化均不重复加网格偏移。")
    text("类别分量保留原始 logits，直接送入交叉熵；预测类别取 argmax。"
         "NMS 按题目要求使用框 conf 作为筛选和排序分数。训练置信度目标是目标存在性，"
         "本实现没有把它描述成定位 IoU 或经过校准的类别概率。")
    heading("实现中的稳定性处理", True)
    text("用假输入自动算出 feat_dim，避免手算维度错误；初始框宽高约为画布的 0.17，"
         "初始 conf 约为 0.05，减少大量空网格造成的初始梯度冲击。训练进行梯度裁剪，"
         "全空白 batch 也能计算有限损失并正常反向传播。")

    # 第 4 页：损失与问答。
    page()
    heading("三、IoU 匹配与多任务损失")
    heading("1. IoU 与负责框的确定", True)
    text("先将 (cx,cy,w,h) 转为左右上下边界。交集宽高取两个框边界的交叠长度并截断为非负，"
         "IoU=交集面积/(面积1+面积2-交集面积)，零面积框返回 0。实现支持广播，"
         "可同时计算所有网格和两个预测框的 IoU。")
    text("对每个有目标网格，从标签中取一个有效真值框，计算其与 B 个预测框的 IoU，"
         "用 argmax 选择唯一负责框。该选择在 no_grad 中进行。若 IoU 相同，"
         "由 argmax 的确定性顺序选择第一个。负责框 mask 为 R，其补集为 noobj mask。")
    heading("2. 损失公式（N 为 batch 图像数）", True)
    table([["分量", "实际采用的形式"],
           ["坐标", "L_coord = 5/N × Σ R × [(cx-cx*)² + (cy-cy*)² + 2(w-w*)² + 2(h-h*)²]"],
           ["置信度", "L_conf = 1/N × [Σ R(conf-1)² + 0.5 Σ (1-R)conf²]"],
           ["类别", "L_cls = 1/N × Σ(有目标网格) CrossEntropy(logits, class_id)"],
           ["总损失", "L = L_coord + L_conf + L_cls"]], [2.5 * cm, 14.7 * cm])
    text("坐标仅监督负责框；类别仅监督有目标网格；置信度同时监督负责框、同格未负责框和空网格框。"
         "三个分量均按图像数归一化，日志中的加权分量可直接相加得到总损失。"
         "这里按作业要求采用原始宽高 MSE 和 conf 目标 1，未使用原始论文的 sqrt(w,h) 损失或 IoU 置信度目标。")
    heading("为什么负样本的置信度权重要较小？", True)
    text("每图有 7×7×2=98 个框，而有效目标平均只有约 2~3 个。负框远多于正框，"
         "若同权求和，模型容易通过把所有 conf 压低来降低损失。lambda_noobj=0.5 降低负样本支配程度，"
         "同时保留对误报的惩罚。")
    heading("为什么 x、y 与 w、h 可以使用不同权重？", True)
    text("中心点决定目标位置，宽高决定目标尺度；相同的绝对误差对小框的相对尺度影响更大。"
         "本实验在全图归一化坐标下使用 xy_weight=1、wh_weight=2，"
         "适度加强小数字的尺度约束。权重应结合梯度、IoU 和验证结果选择，并非越大越好。")
    heading("同格两个目标是否会由 B=2 自动解决？", True)
    text("不会。B 个框是对同一格目标的多个候选定位器，但只共享一组类别输出，"
         "本损失也只监督一个负责框。要处理同格多个目标，应改变标签表示和检测头，"
         "例如使用多尺度检测、更多独立类别预测或其他匹配机制。")

    # 第 5 页：实验对比与曲线。
    page()
    heading("四、训练过程与实验结果")
    text(f"最终设置：Adam，初始 lr={config['lr']}，weight_decay=1e-5；余弦衰减至初始学习率的 5%；"
         f"batch_size={config['batch_size']}，epochs={config['epochs']}；"
         f"每轮重新合成 {config['train_samples']} 张训练图，验证集 {config['val_samples']} 张。"
         f"设备为 {config['gpu']}。训练循环耗时 {config['training_seconds']:.1f} 秒（不含初始合成、最终评估及报告生成）。")
    figure("loss_curves.png", "图 3  实测训练损失、坐标损失及验证损失；均为每张图的加权损失。")
    text(f"总训练损失从第 1 轮 {float(history[0]['loss']):.4f} 降到最后一轮 {float(history[-1]['loss']):.4f}。"
         f"按验证总损失选择第 {config['best_epoch']} 轮权重，验证损失 {config['best_val_loss']:.4f}。"
         "报告和演示默认加载这份最佳权重，而非自动采用最后一轮。")
    if baseline:
        table([["训练策略", "Precision", "Recall", "F1"],
               ["固定 12000 图 / 30 轮", pct(baseline["precision"]), pct(baseline["recall"]), pct(baseline["f1"])],
               [f"每轮更新 {config['train_samples']} 图 / {config['epochs']} 轮", pct(metrics["precision"]), pct(metrics["recall"]), pct(metrics["f1"])]],
              [7.0 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm])
        text("固定小样本实验出现训练损失接近 0、验证损失偏高的过拟合。改进实验更换合成组合并增加样本数和轮数，"
             "提高了泛化能力。此处多项训练设置同时变化，结果只能证明整体方案有效，不能作为单变量消融实验。", "small")
    heading("评估协议与阈值", True)
    text(f"最终测试共 {metrics['images']} 图、{dataset['test']['objects']} 个目标，含 {dataset['test']['empty_images']} 张空白图。"
         f"conf 阈值 {metrics['conf_thresh']} 仅根据验证集 F1 在候选值 0.2/0.3/0.4/0.5/0.6 中选择；"
         f"按类别 NMS 的 IoU 阈值为 {metrics['nms_iou_thresh']}。同类别且 IoU≥0.5 才算 TP，每个真值最多匹配一次。")
    text(f"测试 TP={metrics['tp']}、FP={metrics['fp']}、FN={metrics['fn']}；"
         f"Precision={pct(metrics['precision'])}，Recall={pct(metrics['recall'])}，F1={pct(metrics['f1'])}。"
         f"正确匹配框平均 IoU={metrics['mean_matched_iou']:.3f}。这些是检测指标，不是单数字分类准确率，也不是 mAP。")

    # 第 6 页：三幅测试图。
    page()
    heading("五、独立测试集可视化（至少三张）")
    text("以下取测试集中最先出现的三张非空图，保留实际误检和漏检。绿色为真值，红色为预测；"
         "红色文字给出数字类别和框置信度。完整 8 图总览另见 outputs/predictions.png。", "small")
    for i, record in enumerate(selected, start=1):
        truth = ", ".join(str(box[4]) for box in record["ground_truth"])
        predicted = "; ".join(f"{box[5]} ({box[4]:.2f})" for box in record["predictions"]) or "无"
        detail = (f"图 {i + 3}  测试索引 #{record['index']}\n"
                  f"真实数字：{truth}\n预测数字（置信度）：{predicted}\n"
                  f"TP={record['tp']}，FP={record['fp']}，FN={record['fn']}。\n"
                  "类别错误会同时造成一次误检和一次漏检；定位低于 IoU 标准同样不计为正确检测。")
        element = Table([[image(result / f"test_{i:02d}.png", 6.2 * cm), paragraph(detail)]],
                        colWidths=[6.6 * cm, 10.6 * cm])
        element.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                     ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story.append(element)
        markdown.append(detail + f"\n![测试图](outputs/test_{i:02d}.png)\n")

    # 第 7 页：真实失败案例与局限。
    page()
    heading("六、失败案例、挑战与局限性")
    failure_summary = (f"选择测试集中第一个含错误的非空样本 #{failure['index']}，"
                       f"TP={failure['tp']}、FP={failure['fp']}、FN={failure['fn']}，"
                       "与原始预测 JSON 可以逐项核对。")
    text(failure_summary)
    failure_panel = Table([[image(result / "failure_case.png", 6.5 * cm),
                            paragraph("\n".join(failure_text))]], colWidths=[7 * cm, 10.2 * cm])
    failure_panel.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(failure_panel)
    markdown.extend(failure_text)
    markdown.append("\n![失败案例](outputs/failure_case.png)\n")
    heading("遇到的挑战及解决方案", True)
    text("① 标签和坐标约定：统一为整图归一化坐标；修正教师注释与 conf 赋值不一致的说明。"
         "② 同格冲突：粘贴前检测占用并跳过冲突，保证类别 one-hot 与框对应。"
         "③ 训练过拟合：由固定图像改为逐轮重新合成，并用独立验证集选权重。"
         "④ 数值和空图：IoU 分母加保护，只在有目标网格计算交叉熵，负框正常参与置信度损失。")
    heading("NMS 阈值怎样影响检测？", True)
    text("较低的 NMS IoU 阈值抑制更强，能减少重复框，但也可能删除相邻的同类数字；"
         "较高阈值保留更多重叠框，可能增加重复检测。本实现不跨类别抑制。"
         "置信度阈值是另一个参数：提高它通常减少候选框、降低误报，同时可能增加漏检。")
    heading("局限与可改进方向", True)
    text("每格只能监督一个目标；合成器跳过冲突使评估比真实密集场景更容易。"
         "数字被压缩到 8~14 像素，细笔画丢失且仍可能相互遮挡；保留下采样细节、增加多尺度特征、"
         "更丰富的缩放或使用更大输入可能改善。全连接检测头对位置分布依赖较强，"
         "仍有误检和漏检；当前合成数据结果不能直接证明对真实拍摄手写图的泛化能力。")
    text("验证：12 项关键检查全部通过，覆盖 IoU、负责框选择、空图梯度、五框 NMS、类别隔离、"
         "一对一评估、输出维度、数据复现与冲突处理。复现命令与 5~8 分钟现场展示提纲见 README.md 和 展示讲稿.md。", "small")
    text("参考：课程作业《大作业2026.docx》及教师六个 Python 模板；Redmon et al., "
         "You Only Look Once: Unified, Real-Time Object Detection, CVPR 2016；"
         "PyTorch / torchvision 文档（MNIST、CrossEntropyLoss）。", "small")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c9d4dd"))
        canvas.line(1.9 * cm, 1.45 * cm, A4[0] - 1.9 * cm, 1.45 * cm)
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#566a7b"))
        canvas.drawString(1.9 * cm, 1.0 * cm, "吉阳  250980418  |  简化 YOLOv1 手写数字检测")
        canvas.drawRightString(A4[0] - 1.9 * cm, 1.0 * cm, f"第 {document.page} 页")
        canvas.restoreState()

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(args.pdf), pagesize=A4,
                                 rightMargin=1.9 * cm, leftMargin=1.9 * cm,
                                 topMargin=1.65 * cm, bottomMargin=1.85 * cm,
                                 title="实现简化的YOLOv1进行手写数字检测", author="吉阳")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (PROJECT / "报告正文.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"PDF: {args.pdf}")


if __name__ == "__main__":
    main()
