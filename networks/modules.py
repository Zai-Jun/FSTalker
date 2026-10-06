import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Convolution => BN => ReLU) * 2
    (卷积 => BN => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    使用最大池化下采样后接双卷积
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling with skip connection then double conv
    使用跳跃连接上采样后接双卷积
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # in_channels is concatenated feature: (skip + upsampled), so mid_channels is set to in_channels // 2
            # in_channels 是拼接特征：(跳跃连接 + 上采样输出)，所以 mid_channels 设为 in_channels // 2
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Compute spatial differences and pad
        # 计算空间尺寸差异并填充
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        # 沿通道维度拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class FourierEncoding3D(nn.Module):
    """
    Apply Fourier feature encoding to a 3-channel position map.
    对三通道位置图应用傅里叶特征编码。
    Input: [B, 3, H, W]
    输入：[B, 3, H, W]
    Output: [B, 3 + 6 * num_bands, H, W], where for each of the 3 channels we
    append sin(2π f_k x_c) and cos(2π f_k x_c) for k=1..num_bands.
    输出：[B, 3 + 6 * num_bands, H, W]，其中对每个通道附加 sin(2π f_k x_c) 和 cos(2π f_k x_c)，k=1..num_bands。
    """
    def __init__(self, num_bands=6, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        # generate log-spaced frequencies from 1 to max_freq
        # 生成从 1 到 max_freq 的对数间隔频率
        self.register_buffer('freqs', torch.logspace(0.0, torch.log10(torch.tensor(max_freq)),
                                                     steps=num_bands))

    def forward(self, pos3d: torch.Tensor):
        """
        pos3d: [B, 3, H, W]
        pos3d：[B, 3, H, W]
        returns: [B, 3 + 6*num_bands, H, W]
        返回：[B, 3 + 6*num_bands, H, W]
        """
        B, C, H, W = pos3d.shape
        # expand to [B, 3, num_bands, H, W]
        # 扩展为 [B, 3, num_bands, H, W]
        p = pos3d.unsqueeze(2) * self.freqs.view(1, 1, self.num_bands, 1, 1) * 2 * torch.pi
        sin = p.sin()
        cos = p.cos()
        sin = sin.view(B, 3 * self.num_bands, H, W)
        cos = cos.view(B, 3 * self.num_bands, H, W)

        return torch.cat([pos3d, sin, cos], dim=1)