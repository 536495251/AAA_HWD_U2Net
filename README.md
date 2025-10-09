# AAA_HWD_U2Net

> 红外小目标检测（Infrared Small-Target Detection）专用 U²-Net 变体集合，融合小波下采样（HWD）与自适应注意力（ACAL / ASAL / CSAF），三个版本（V0 / V1 / V2）代表设计演进：从基础实现 → 空间/通道注意力改进 → 深浅层空间注意力与更细粒度的自适应聚合。

---

## 目录

* [简介](#简介)
* [快速开始](#快速开始)
* [架构总览](#架构总览)
* [关键模块详解](#关键模块详解)
* [模块差异与改进点](#模块差异与改进点)
* [性能指标](#性能指标)
* [使用示例](#使用示例)
* [文件结构](#文件结构)
* [许可证 & 致谢](#许可证--致谢)

---

# 简介

AAA_HWD_U2Net 是基于 U²-Net 思路并针对红外小目标检测做定制化改造的系列模型。核心设计思路是：

1. **小波下采样（HWD）**：在下采样过程中保留高频/细节信息，减少目标信息损失；
2. **自适应通道聚合（ACAL）**：增强通道特征相关性，抑制冗余信息；
3. **自适应空间聚合（ASAL）与通道-空间融合（CSAF）**：通过空间注意力增强目标响应，抑制背景干扰；
4. **深浅层空间注意力策略（V2）**：浅层聚焦细节，深层捕获语义，提高小目标检测精度。

仓库中包含三个版本（V0 / V1 / V2），从基础到优化逐步增强检测能力和稳定性。

---

# 快速开始

依赖：

```bash
Python >= 3.8
PyTorch >= 1.10
pytorch_wavelets
thop
```

示例加载：

```python
from models.my_model.AAA_HWD_U2Net_V2 import AAA_HWD_U2Net_V2
model = AAA_HWD_U2Net_V2(out_ch=1)
model.eval()
```

---

# 架构总览

* **Encoder / Decoder** 基于 RSU（Residual U-block）与 RSU4F（膨胀卷积版）子模块，形成 U-in-U 结构。
* **HWD 小波下采样**：前两层编码器优先使用，保留高频细节。
* **解码器**：每层先插值对齐，再通过 CSAF（通道+空间融合）增强特征。
* **侧分支输出**：深度监督训练，融合输出用于推理。

---

# 关键模块详解

* **ConvBNReLU / DownConvBNReLU / UpConvBNReLU**：卷积+BN+ReLU，构建 RSU 基础结构。
* **RSU / RSU4F**：残差 U-block，RSU4F 不下采样，使用膨胀卷积保持输入分辨率。
* **HWD（Haar Wavelet Downsampling）**：小波分解获得低频+高频，1x1卷积压缩通道。
* **ACAL**：自适应通道聚合，通过 1D 卷积自调整层数和核大小。
* **ASAL**：自适应空间聚合，指数膨胀卷积实现大感受野，生成空间注意力图。
* **ChannelAttention / SpatialAttention / ShallowSpatialAttention / DeepSpatialAttention**：通道/空间注意力分支，V2 将浅层与深层分开处理。
* **CSAF**：融合通道与空间注意力，增强解码器特征响应。
* **Side outputs**：多尺度输出用于深度监督，推理时输出融合主结果。

---

# 模块差异与改进点

## V0

* 基础 HWD + ACAL + ASAL + CSAF 架构
* 单一空间注意力模块
* 优点：保持细节，小目标保真度高

## V1

* ASAL 感受野优化，ChannelAttention 引入 LayerNorm
* SpatialAttention 优化 softmax 归一化与上下文提取

## V2

* 引入 **ShallowSpatialAttention + DeepSpatialAttention**，按层选择不同空间策略
* 解码层自适应选择深浅层注意力，权重 0.4/0.6 平衡浅层细节与深层语义

---

# 性能指标

| Train Dataset | Validate Dataset | Ratio |      Model       | IoU (%) | nIoU (%) | Fa (×10⁻⁶) | Pd (%) | FLOPs（G） | Params (M) |    Epochs    |
|:-------------:|:----------------:|:-----:|:----------------:|:-------:|:--------:|:----------:|:------:|:--------:|:----------:|:------------:|
|   IRSTD-1K    |     IRSTD-1K     |  4:1  | AAA_HWD_U2Net_V0 |  73.18  |  71.18   |    9.88    | 95.71  |  89.36   |    4.11    | best:541/600 |
|   IRSTD-1K    |    NUDT-SIRST    |  4:1  | AAA_HWD_U2Net_V0 |  45.50  |  50.75   |    3.19    | 68.34  |  89.36   |    4.11    | best:541/600 |
|  NUDT-SIRST   |    NUDT-SIRST    |  1:1  | AAA_HWD_U2Net_V0 |  72.86  |  76.62   |    8.92    | 96.54  |  89.36   |    4.11    | best:470/600 |
|  NUAA-SIRST   |    NUAA-SIRST    |  4:1  | AAA_HWD_U2Net_V0 |  76.84  |  69.97   |   34.75    | 98.94  |  89.36   |    4.11    | best:464/600 |
|   IRSTD-1K    |     IRSTD-1K     |  4:1  | AAA_HWD_U2Net_V1 |  72.55  |  70.40   |   10.71    | 94.70  |  86.93   |    4.07    | best:668/800 |
|   IRSTD-1K    |     IRSTD-1K     |  4:1  | AAA_HWD_U2Net_V2 |  72.69  |  70.29   |    9.27    | 95.36  |  89.24   |    4.08    | best:599/600 |
|  NUDT-SIRST   |    NUDT-SIRST    |  1:1  | AAA_HWD_U2Net_V2 |  73.45  |  76.85   |    6.62    | 97.92  |  89.24   |    4.08    | best:599/600 |
|  NUAA-SIRST   |    NUAA-SIRST    |  4:1  | AAA_HWD_U2Net_V2 |  74.31  |  69.48   |   20.55    | 95.74  |  89.24   |    4.08    | best:599/600 |

**分析**：

* V0 IoU 最高，但 Fa 略高
* V1 平滑性最佳，Pd 相对略低
* V2 在保证 Pd 高的同时 Fa 最低，小目标检出更可靠

---

# 使用示例

```python
import torch
from models.my_model.AAA_HWD_U2Net_V2 import AAA_HWD_U2Net_V2

model = AAA_HWD_U2Net_V2(out_ch=1)
x = torch.randn(1, 3, 256, 256)
y = model(x)
print(y.shape)  # [1,1,256,256]

# 训练模式
model.train()
outs = model(x)   # 主输出 + 侧输出列表
```

---

# 文件结构

* AAA_HWD_U2Net_V0.py
* AAA_HWD_U2Net_V1.py
* AAA_HWD_U2Net_V2.py

---

# 许可证 & 致谢

建议使用 MIT 或 Apache-2.0 协议。
引用方式：

```
@article{AAA_HWD_U2Net,
  title={AAA-HWD-U2Net: Infrared Small Target Detection  Adaptive Attention Aggregation and Haar Wavelet Downsampling},
  author={Zhang. Zhen. et al.},
  year={2025},
  journal={Under Review}
}
```
