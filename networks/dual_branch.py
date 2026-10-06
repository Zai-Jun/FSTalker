import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as T
from pytorch3d.transforms import axis_angle_to_quaternion
from networks.modules import (
    DoubleConv,
    Down,
    Up,
    FourierEncoding3D
)

class DetailBlock(nn.Module):
    """
    A small convolutional block to enhance local high-frequency details.
    一个小型卷积块，用于增强局部高频细节。
    """
    def __init__(self, in_ch, expr_dim, hidden_ch=64):
        super().__init__()
        self.expr_proj = nn.Sequential(
            nn.Linear(expr_dim, hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(in_ch + in_ch + hidden_ch, hidden_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(hidden_ch, hidden_ch, 1)

    def forward(self, global_feat, disp_map, expr, mask):
        """
        global_feat: [B, C, H, W]
        disp_map:    [B, C, H, W]
        expr:        [B, expr_dim]
        mask:        [1, 1, H, W]  (binary mask)
        global_feat: [B, C, H, W]  全局特征
        disp_map:    [B, C, H, W]  位移图
        expr:        [B, expr_dim] 表情参数
        mask:        [1, 1, H, W]  二值掩码
        """
        B, C, H, W = global_feat.shape

        e = self.expr_proj(expr).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        disp_map_masked = disp_map * mask
        global_feat_masked = global_feat * mask

        x = torch.cat([disp_map_masked, global_feat_masked, e], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv_out(x)

        return x * mask


class DualBranchUNet(nn.Module):
    """
    Dual-branch U-Net: global branch + local detail branch.
    双分支 U-Net：全局分支 + 局部细节分支。
    """
    def __init__(
        self,
        device,
        uv_sample_coords,
        uv_mask,
        reference_image=None,
        position_map=None,
        condition_dim=118,
        bilinear=True,
        uv_size=256,
        head_mask=None,
    ):
        super().__init__()
        self.device = device
        self.uv_size = uv_size
        self.eye_mask = uv_mask["eye_region"]
        self.nose_mask = uv_mask["nose"]
        self.lips_mask = uv_mask["lips"]
        self.forehead_mask = uv_mask["forehead"]

        self.transform = T.Compose([T.Resize((uv_size, uv_size), antialias=True), T.ToTensor()])

        self.resize = T.Resize((uv_size, uv_size), antialias=True)

        if reference_image is not None:
            self.register_buffer("reference_image", self.transform(reference_image).unsqueeze(0).detach().to(device))
        else:
            self.register_buffer("reference_image", torch.empty(1, 3, uv_size, uv_size))

        if position_map is not None:
            self.register_buffer("position_map", position_map.detach().float().to(device))
        else:
            self.register_buffer("position_map", torch.empty(1, 3, uv_size, uv_size))

        if head_mask is not None:
            self.register_buffer("head_mask", head_mask.detach().float().to(device))
        else:
            self.head_mask = None

        self.uv_sample_coords = uv_sample_coords.detach().float().to(device)
        self.condition_dim = condition_dim
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        #=========== Encoder ===========
        #=========== 编码器 ===========
        self.inc = DoubleConv(3, 64, mid_channels=32)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)

        #=========== Process different features ===========
        #=========== 处理不同特征 ===========
        self.fourier = FourierEncoding3D(num_bands=6, max_freq=10.0)
        self.uv_conv = nn.Sequential(nn.Conv2d(3 + 6 * 6, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.disp_conv = nn.Sequential(nn.Conv2d(3 + 6 * 6, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.expr_embed = nn.Sequential(nn.Linear(100, 96), nn.ReLU(inplace=True))
        self.pose_embed = nn.Sequential(nn.Linear(15, 16), nn.ReLU(inplace=True))
        self.tran_embed = nn.Sequential(nn.Linear(3, 16), nn.ReLU(inplace=True))

        #=========== Global Decoder ============
        #=========== 全局解码器 ============
        self.global_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        #=========== Local Decoder ============
        #=========== 局部解码器 ============
        self.local_decoder = DetailBlock(in_ch=64, expr_dim=100, hidden_ch=64)

        #=========== Fuse Layer ===========
        #=========== 融合层 ===========
        self.final_fuse = nn.Sequential(
            nn.Conv2d(128 + 4 * 64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 13, kernel_size=1),
        )

        self._initialize_weights()

    #===========================================================================
    # 阶段二 - 子模块二：双分支特征场生成 (Feature Field Generation)
    # 
    # 输入 (4 路信号)：
    #   Img_r       : 身份参考纹理图 (reference_image)     - [1, 3, 256, 256]
    #   UV_pos      : UV 位置图 (position_map)             - [1, 3, 256, 256]
    #   UV_disp     : UV 位移图 (displacement_map)         - [1, 3, 256, 256]
    #   β (控制码)  : FLAME 表情/姿态参数 (expr 100维 + pose 15维 + trans 3维)
    #
    # 输出：
    #   Δ_map       : UV 特征偏移图                         - [1, N, 13]
    #     13 维分解：
    #       offset[:, 0:3]   → 位置偏移 (Δx, Δy, Δz)
    #       offset[:, 3:6]   → 缩放射频 (Δsx, Δsy, Δsz)
    #       offset[:, 6:10]  → 旋转偏移 (四元数 Δr, Δx, Δy, Δz)
    #       offset[:, 10:13] → 颜色偏移 (Δr, Δg, Δb)
    #       offset[:, 13]    → 透明度偏移 (Δα)
    #
    # 网络结构：
    #   UNet 编码器 (Ref Image) → 共享特征 d3
    #     ├── 全局分支 (Global Decoder): d3 + UV_pos(Fourier) + expr + pose + trans → offset_global
    #     ├── 局部分支 (Local Decoder):  对眼/鼻/唇/额四个区域的 displacement_map 做 DetailBlock
    #     │    每个 DetailBlock 输入 d3 + UV_disp(Fourier) + expr + 区域mask
    #     └── 融合层 (Fuse): concat(offset_global, local_eye, local_nose, local_lips, local_forehead)
    #                        → Conv2d → 13通道偏移图
    #===========================================================================
    def forward(
        self, flame_param, timestep, displacement_map, return_map=False
    ):
        # process condition
        # 处理条件输入
        expr = flame_param["expr"][timestep].unsqueeze(0)  # 1*100
        neck_pose = flame_param["neck_pose"][timestep].unsqueeze(0)  # 1*3
        jaw_pose = flame_param["jaw_pose"][timestep].unsqueeze(0)  # 1*3
        eyes_pose = flame_param["eyes_pose"][timestep].unsqueeze(0)  # 1*6
        rotation = flame_param["rotation"][timestep].unsqueeze(0)  # 1*3
        pose = torch.cat((neck_pose, jaw_pose, eyes_pose, rotation), dim=1)
        translation = flame_param["translation"][timestep].unsqueeze(0)  # 1*3

        #========== Encoder ==========
        #========== 编码器 ==========
        x1 = self.inc(self.reference_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)

        #========== Global Decoder ==========
        #========== 全局解码器 ==========
        pos_fourier = self.fourier(self.position_map.float())
        uv_feat = self.uv_conv(pos_fourier)
        expr_emb = self.expr_embed(expr)
        expr_feat = (expr_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))
        pose_emb = self.pose_embed(pose)
        pose_feat = (pose_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))
        trans_emb = self.tran_embed(translation)
        trans_feat = (trans_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3)))
        global_feat = torch.cat([d3, uv_feat, expr_feat, pose_feat, trans_feat], dim=1)
        offset_global = self.global_decoder(global_feat)

        #========== Local Decoder ==========
        #========== 局部解码器 ==========
        disp_f = self.fourier(displacement_map.float())
        disp = self.disp_conv(disp_f)

        local_eye = self.local_decoder(d3, disp, expr, self.eye_mask)
        local_nose = self.local_decoder(d3, disp, expr, self.nose_mask)
        local_lips = self.local_decoder(d3, disp, expr, self.lips_mask)
        local_forehead = self.local_decoder(d3, disp, expr, self.forehead_mask)

        #========= Fuse Layer =========
        #========= 融合层 =========
        unified_feat = torch.concat(
            [offset_global, local_eye, local_nose, local_lips, local_forehead], dim=1
        )
        offset_out = self.final_fuse(unified_feat)
        offset_final = self._offset_attr_process(offset_out, self.uv_sample_coords)

        if return_map:
            return offset_final, offset_out
        return offset_final

    def forward_once(self):
        # for testing, only calculate d3 once
        # 测试时使用，仅计算一次 d3 特征
        x1 = self.inc(self.reference_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        self.d3 = self.up3(d2, x1)
        pos_fourier = self.fourier(self.position_map.float())
        self.uv_feat = self.uv_conv(pos_fourier)


    def _decode(self, flame_param, timestep, displacement_map):
        expr = flame_param["expr"][timestep].unsqueeze(0)  # 1*100
        neck_pose = flame_param["neck_pose"][timestep].unsqueeze(0)  # 1*3
        jaw_pose = flame_param["jaw_pose"][timestep].unsqueeze(0)  # 1*3
        eyes_pose = flame_param["eyes_pose"][timestep].unsqueeze(0)  # 1*6
        rotation = flame_param["rotation"][timestep].unsqueeze(0)  # 1*3
        pose = torch.cat((neck_pose, jaw_pose, eyes_pose, rotation), dim=1)
        translation = flame_param["translation"][timestep].unsqueeze(0)  # 1*3

        #========= Global Decoder ===========
        #========= 全局解码器 ===========
        H, W = self.d3.shape[2], self.d3.shape[3]
        pos_fourier = self.fourier(self.position_map.float())
        uv_feat = self.uv_conv(pos_fourier)
        expr_emb = self.expr_embed(expr)
        expr_feat = (
            expr_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        pose_emb = self.pose_embed(pose)
        pose_feat = (
            pose_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        trans_emb = self.tran_embed(translation)
        trans_feat = (
            trans_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )
        global_feat = torch.cat([self.d3, uv_feat, expr_feat, pose_feat, trans_feat], dim=1)
        offset_global = self.global_decoder(global_feat)

        #========= Local Decoder ==========
        #========= 局部解码器 ==========
        disp_f = self.fourier(displacement_map.float())
        disp = self.disp_conv(disp_f)

        local_eye = self.local_decoder(self.d3, disp, expr, self.eye_mask)
        local_nose = self.local_decoder(self.d3, disp, expr, self.nose_mask)
        local_lips = self.local_decoder(self.d3, disp, expr, self.lips_mask)
        local_forehead = self.local_decoder(self.d3, disp, expr, self.forehead_mask)

        #========= Fuse Layer ==========
        #========= 融合层 ==========
        unified_feat = torch.concat(
            [offset_global, local_eye, local_nose, local_lips, local_forehead], dim=1
        )
        offset_out = self.final_fuse(unified_feat)
        offset_final = self._offset_attr_process(offset_out, self.uv_sample_coords)

        return offset_final

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def update_uv_coords(self, uv_sample_coords):
        self.uv_sample_coords = uv_sample_coords.detach().float().to(self.device)

    #===========================================================================
    # 阶段二 - 子模块三：UV 自适应采样与属性偏移处理
    #
    # 将双分支 U-Net 输出的 UV 特征偏移图 Δ_map (13通道)，按每个高斯基元的 UV 坐标
    # 进行双线性采样（grid_sample），得到每个高斯基元专属的偏移量 δ_i [1, N, 13]。
    # 然后对 13 个维度分别施加物理上有意义的激活约束。
    #
    # 13 维偏移分解：
    #   位置  (0:3)   → tanh × 0.1     范围 [-0.1, 0.1]     (毫米级微调)
    #   缩放  (3:6)   → exp             范围 (0, +∞)         (只能放大)
    #   旋转  (6:10)  → axis_angle→quat 单位四元数           (合法旋转变换)
    #   颜色  (10:13) → tanh × 0.7     范围 [-0.7, 0.7]     (RGB 微调)
    #   透明度(13)    → tanh × 0.5     范围 [-0.5, 0.5]     (透明度微调)
    #===========================================================================
    def _offset_attr_process(self, offset_map, uv_coords):
        """
        Args:
            offset_map: [B, 13, 512, 512]
            uv_coords:  [B, N, 2]
        Returns:
            offset:     [B, N, 13]
        从偏移图中采样每个高斯的属性偏移量
        参数：
            offset_map: [B, 13, 512, 512] 偏移图
            uv_coords:  [B, N, 2] UV 坐标
        返回：
            offset:     [B, N, 13] 每个点的偏移量
        """
        # sample per-point offset from the map
        # 从偏移图中采样每个点的偏移量
        # 核心操作：grid_sample 在每个高斯 UV 位置做双线性插值
        offset = self._uv_look_up(texture=offset_map, uv_coords=uv_coords)

        # apply activations to individual channels
        # 对每个通道分别应用激活函数
        # 不同属性需要不同的值域约束，以确保物理合理性
        return torch.cat(
            [
                self._offset_position_activation(offset[:, :, 0:3]),   # 位置
                self._offset_scaling_activation(offset[:, :, 3:6]),     # 缩放
                self._rotation_activation(offset[:, :, 6:9]),           # 旋转 (轴角→四元数)
                self._offset_color_activation(offset[:, :, 9:12]),      # 颜色
                self._offset_opacity_activation(offset[:, :, 12:13]),   # 透明度
            ],
            dim=-1,
        )

    @staticmethod
    def _offset_position_activation(input):
        """
        Force the NN output serves as position to be between [-0.1, 0.1]
        将网络输出的位置偏移限制在 [-0.1, 0.1] 范围内
        
        位置偏移控制毫米级微调，避免高斯跑飞远离 FLAME 网格面。
        """
        return torch.tanh(input) * 0.1

    @staticmethod
    def _offset_color_activation(input):
        """
        Force the NN output serves as color to be between [-0.7, 0.7].
        将网络输出的颜色偏移限制在 [-0.7, 0.7] 范围内
        
        颜色用 tanh 约束到 [-0.7, 0.7]，与 Gaussian 的 SH0 系数叠加后
        通过 clamp_min(+0.5) 保证最终颜色在 [0, 1] 合法范围内。
        """
        return torch.tanh(input) * 0.7

    @staticmethod
    def _offset_scaling_activation(input):
        """
        Force the NN output serves as scaling within proper range [0, +inf].
        将网络输出的缩放值映射到 [0, +inf) 范围
        
        缩放用 exp 激活，保证为正数。与阶段一 scale 相乘实现逐高斯缩放微调。
        输入 clamp 到 [-5, 5] 防止 exp 溢出。
        """
        return torch.exp(torch.clamp(input, -5, 5))

    @staticmethod
    def _offset_opacity_activation(input):
        """
        Force the NN output serves as scaling within proper range [-0.5, 0.5].
        将网络输出的透明度偏移限制在 [-0.5, 0.5] 范围内
        
        透明度偏移用 tanh 约束到 [-0.5, 0.5]，叠加后 clamp 到 [0, 1]。
        """
        return torch.tanh(input) * 0.5

    def _rotation_activation(self, input: torch.Tensor) -> torch.Tensor:
        """
        Convert NN output to quaternion rotation [r, x, y, z] with proper normalization.
        Supports only [B, N, 3]input.

        Args:
            input: Tensor of shape [B, N, 3] — axis-angle representation.
        Returns:
            Tensor of shape [B, N, 4] — quaternion in [r, x, y, z] format.
        将网络输出转换为四元数旋转 [r, x, y, z]，并进行归一化。
        仅支持 [B, N, 3] 输入（轴角表示）。
        
        旋转偏移的数学处理：
          1. tanh 将输出压缩到 [-π, π]，作为轴角表示的旋转向量
          2. pytorch3d.axis_angle_to_quaternion 转为单位四元数 (w, x, y, z)
          3. 在 render 阶段通过 Hamilton 乘积与阶段一旋转复合
        """
        input = torch.tanh(input) * (torch.pi)  # scale to [-π, π]
        quat = axis_angle_to_quaternion(input)  # [B, N, 4] wxyz

        return quat.contiguous()

    def _uv_look_up(
        self, texture: torch.Tensor, uv_coords: torch.Tensor
    ) -> torch.Tensor:
        """Use sampling to get gs value
        Args:
            texture:    [B, C, H, W] texture color
            uv_coords:  [B, N, 2] texture uv
        Returns:
            values:     [B, N, C] texture sampling
        通过 UV 采样获取高斯属性值
        参数：
            texture:    [B, C, H, W] 纹理图
            uv_coords:  [B, N, 2] UV 坐标
        返回：
            values:     [B, N, C] 采样值
        
        阶段二核心采样操作：
          1. 将 UV 坐标 (0~1) 映射到 grid_sample 的坐标空间 (-1~1)
          2. 通过双线性插值 (bilinear) 从连续纹理图上采样
          3. 每个高斯基元通过自己的 UV 坐标获得专属的属性偏移
          这相当于在 UV 空间实现了 "Soft Binding" — 不强制高斯绑在三角形质心，
          而是让网络在连续 UV 空间上自由预测偏移。
        """
        grid = uv_coords * 2.0 - 1.0  # [B, N, 2]
        grid = grid.unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(
            texture, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        sampled_value = sampled.squeeze(3).permute(0, 2, 1)  # [B, N, C]

        return sampled_value