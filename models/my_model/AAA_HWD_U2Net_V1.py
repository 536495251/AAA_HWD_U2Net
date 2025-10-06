from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from thop import profile
from math import ceil, log2, log

class ConvBNReLU(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)  # 插值上采样
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    """ Residual U-block """

    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)  # stem
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))  # 包含height-1个上采样层
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))  # 添加height-2个中间下采样层
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))  # 最后一个编码层使用膨胀卷积（dilation=2）增加感受野

        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)
        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        # RSU4F使用膨胀卷积而不是下采样来捕获多尺度信息，保持输入分辨率不变
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in

# Adaptive Channel Aggregation Layer
class ACAL(nn.Module):
    def __init__(self, channel, n=2, b=2):
        super(ACAL, self).__init__()
        kernel_size = int(abs((log(channel, 2) + 1) / 2))
        self.kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.layer_size = 1 + (log(channel, 2) - b) / (2 * n)
        self.layer_size = ceil(self.layer_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size,
                              padding=int(self.kernel_size / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.conv(x.squeeze(-1).transpose(-1, -2))
        for _ in range(self.layer_size):
            h = self.conv(h)
        h = h.transpose(-1, -2).unsqueeze(-1)
        h = self.sigmoid(h)
        return x + h


class ASAL(nn.Module):
    """
    Adaptive Spatial Aggregation Layer.
    Uses small kernel (3x3) + exponential dilation (1,2,4,...) to reach a target receptive field.
    Returns x + sigmoid(attn) (residual addition).
    """

    def __init__(self,target=7, base_kernel=3):
        super().__init__()
        assert base_kernel % 2 == 1, "base_kernel must be odd"

        # target receptive field
        R_target =target
        # compute required number of dilation-steps L
        # RF(L) ≈ 2^(L+1) - 1  => L ≈ ceil(log2((R_target+1)/2))
        L = int(ceil(log2((R_target + 1) / 2.0)))
        self.L = max(1, L)  # at least 1 layer
        self.kernel = base_kernel
        self.layers = nn.ModuleList()
        # use small 2D convs with increasing dilation
        for i in range(self.L):
            d = 2 ** i
            padding = d * (self.kernel // 2)
            # input is single-channel spatial map (we aggregate channels before calling)
            self.layers.append(
                nn.Conv2d(1, 1, kernel_size=self.kernel, padding=padding, dilation=d, bias=False)
            )

        self.act = nn.ReLU(inplace=True)  # optional non-linearity between dilated convs
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        pipeline:
          - aggregate channels -> [B, 1, H, W]
          - pass through series of dilated convs (ReLU between them)
          - apply sigmoid -> spatial attention map [B,1,H,W]
          - residual add: output = x + attn
        """
        # channel aggregation (mean); you can also use (avg + max) concat if wanted
        h = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]

        for conv in self.layers:
            h = conv(h)
            h = self.act(h)

        attn = self.sigmoid(h)  # [B,1,H,W]
        return x *(1+attn)  # residual add


class ChannelAttention(nn.Module):
    """通道注意力分支"""

    def __init__(self, in_channel, stride=1, ratio=4):
        super(ChannelAttention, self).__init__()
        inter_channel = in_channel // 2
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, 1, 1, stride, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True)
        )
        self.conv_up = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel // ratio, kernel_size=1),
            nn.LayerNorm([inter_channel // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel // ratio, in_channel, kernel_size=1),
            nn.LayerNorm([in_channel, 1, 1]),
        )
        self.ACAL = ACAL(inter_channel)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, E, D):
        D_ = self.conv_2(D)
        B, C, H, W = D_.size()
        D_ = D_.view(B, C, H * W)
        E = self.conv_1(E).view(B, 1, H * W)
        E = self.softmax(E)
        context = torch.matmul(D_, E.transpose(1, 2)).unsqueeze(-1)  # [B, IC, 1, 1]
        context = self.conv_up(self.ACAL(context))  # [B, C, 1, 1]
        out = D * self.sigmoid(context)
        return out
class SpatialAttention(nn.Module):
    """空间注意力分支"""
    def __init__(self, in_channel, stride=1,ratio=4):
        super(SpatialAttention, self).__init__()
        inter_channel = in_channel // 2
        self.inter_channel=inter_channel
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            #nn.Conv2d(inter_channel, inter_channel, 1, stride, 0, bias=False),
            nn.Conv2d(inter_channel, 1, 1, stride, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.conv_3_ = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(True)
        )
        self.conv_up = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel // ratio, kernel_size=1),
            nn.BatchNorm2d(inter_channel // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel // ratio, in_channel, kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ASAL = ASAL(target=32)
        self.ACAL= ACAL(inter_channel)

    def forward(self, E, D):
        E__=self.conv_3_(E)
        batch, channel, height, width = E__.size()
        E__ = self.ACAL(self.avg_pool(E__))  # [N, IC, 1, 1]
        batch, channel, avg_e_h, avg_e_w = E__.size()
        E__ = E__.view(batch, channel, avg_e_h * avg_e_w).permute(0, 2, 1)  # [N, 1, IC]
        E__ = self.softmax(E__)
        D__ = self.conv_4(D).view(batch, self.inter_channel, height * width)  # [N, IC, H*W]
        context__ = torch.matmul(E__, D__).view(batch, 1, height, width)  # [N, 1, H, W]
        context__ = F.layer_norm(context__, normalized_shape=(1, context__.shape[-2], context__.shape[-1]))
        out__ = E * self.sigmoid(context__)  # [N, 1, H, W]
        E_ = self.conv_3(E)  # [B, 1, H, W]
        D_ = self.conv_4(D)  # [B, C, H, W]
        B, _, H, W = E.shape
        # 展开为序列
        E_vec = E_.view(B, 1, H * W)  # [B,1,HW]
        E_vec = F.softmax(E_vec, dim=-1)  # [B,1,HW]
        D_vec = D_.view(B, -1, H * W)  # [B,C',HW]
        # 计算空间 context
        context = torch.matmul(D_vec, E_vec.transpose(1, 2))  # [B,C',1]
        context = context.view(B, -1, 1, 1)  # [B,C',1,1]
        # ASAL 细化空间特征（自适应空间建模）
        context = context.expand(-1, -1, H, W)  # [B,C',H,W]
        context = self.ASAL(context)  # [B,C',H,W]
        # 升维成 mask
        attn_mask = self.conv_up(context)  # [B,1,H,W]
        # 作用到 D 或 E（可选）
        out_ = D * self.sigmoid(attn_mask)
        return out_*0.4+out__*0.6

class CSAF(nn.Module):
    """融合 ChannelAttention 和 SpatialAttention """

    def __init__(self, in_channel, stride=1):
        super(CSAF, self).__init__()
        self.channel_att = ChannelAttention(in_channel, stride)
        self.spatial_att = SpatialAttention(in_channel, stride)
        self.out = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, 3, 1, 1),
            nn.BatchNorm2d(in_channel * 2),
            nn.ReLU(True)
        )

    def forward(self, E, D):
        chn = self.channel_att(E, D)
        spa = self.spatial_att(E, D)
        out = self.out(chn + spa)
        return out

# 小波下采样模块
class HWD(nn.Module):
    def __init__(self, in_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 小波分解后通道数变 4 倍，这里用 1x1 conv 压回 in_ch
        self.reduce = nn.Conv2d(in_ch * 4, in_ch, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = x.float()  # 强制输入为 float32
            yL, yH = self.wt(x)
            y_HL = yH[0][:, :, 0, :, :]
            y_LH = yH[0][:, :, 1, :, :]
            y_HH = yH[0][:, :, 2, :, :]
            x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # (B, 4C, H/2, W/2)
            x = self.reduce(x)  # 压回 (B, C, H/2, W/2)
            x = self.bn(x)
            x = self.relu(x)
        return x
class AAA_HWD_U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" and "decode" in cfg
        self.encode_num = len(cfg["encode"])
        # 编码器
        encode_list = []
        side_list = []
        pool_list = []
        for i, c in enumerate(cfg["encode"]):
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            if i < 2:  # 前两层
                pool_list.append(HWD(c[3]))
            else:
                pool_list.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)
        self.pool_modules = nn.ModuleList(pool_list)
        # 解码器 + csaf
        decode_list = []
        csaf_list = []
        for c in cfg["decode"]:
            assert len(c) >= 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            csaf_list.append(CSAF(int(c[1] / 2)))
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))

        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.csaf_modules = nn.ModuleList(csaf_list)

        # 最终输出卷积（直接拼接所有 side 分支）
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        # 编码器
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = self.pool_modules[i](x)

        # 解码器
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in zip(self.decode_modules, self.csaf_modules):
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m[1](x, x2)
            x = m[0](x)
            decode_outputs.insert(0, x)
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            return [x] + side_outputs
        else:
            return self.bn(x)


def AAA_HWD_U2Net_V1(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 32, 64, False, False],  # En3
                   [4, 64, 32, 128, False, False],  # En4
                   [4, 128, 32, 128, True, False],  # En5
                   [4, 128, 64, 128, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 256, 64, 128, True, True],  # De5
                   [4, 256, 32, 64, False, True],  # De4
                   [5, 128, 32, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return AAA_HWD_U2Net(cfg, out_ch)


if __name__ == '__main__':
    model = AAA_HWD_U2Net_V1(out_ch=1)
    # 模式设置
    training_mode = True  # True 训练模式，False 推理模式
    model.train() if training_mode else model.eval()
    # 模拟输入数据
    input_tensor = torch.randn(1, 3, 512, 512)  # Batch=1, RGB图像
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
    flops, params = profile(model, (input_tensor,))
    # 输出信息
    if training_mode:
        print(f"Training Mode Output (List of {len(output)} tensors):")
        print("-" * 50)
        print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
        print('Params = ' + str(params / 1000 ** 2) + ' M')
        for i, out in enumerate(output):
            print(f"  Output[{i}] shape: {out.shape}")
    else:  print(f"Inference Mode Output shape: {output.shape}")