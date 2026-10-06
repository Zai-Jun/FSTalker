import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y)

        # 1. 动态获取当前真实的特征维度，防止 hardcode 的 seq_len 带来动态输入截断错误
        current_seq_len = y.numel()

        # 2. 将输入展平并确保内存物理连续
        out = y.contiguous().view(1, current_seq_len)

        # 3. 核心隔离区：解构 attentionNet，用纯张量操作替换全连接层，彻底绕过 cuBLAS 崩溃
        for layer in self.attentionNet:
            if isinstance(layer, torch.nn.Linear):
                # 原始逻辑：out = F.linear(out, layer.weight, layer.bias) -> 会调用崩溃的 cuBLAS
                # 替换逻辑：使用基础矩阵空间广播进行纯数学模拟

                # out 的形状从 [1, in_features] 变为 [1, in_features, 1]
                # weight.T 的形状从 [in_features, out_features] 变为 [1, in_features, out_features]
                w = layer.weight.T.unsqueeze(0)

                # Element-wise 相乘后在 in_features 维度求和，结果严格等价于矩阵乘法 [1, out_features]
                out = torch.sum(out.unsqueeze(2) * w, dim=1)

                # 若存在偏置项，进行广播相加
                if layer.bias is not None:
                    out = out + layer.bias.unsqueeze(0)
            else:
                # 遇到其他层（如 Softmax, ReLU、Dropout 等）正常前向传播
                out = layer(out)

        # 4. 恢复为最终参与音频特征加权相乘的维度
        y = out.view(1, current_seq_len, 1)

        return torch.sum(y * x, dim=1)  # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)

        return x


# ==================================================================================
# Main Motion Network (Refactored for MoE & TimeFormer Control)
# ==================================================================================

class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim=32,
                 ind_dim=0,
                 args=None,
                 # [Debugging Switch]
                 face_sees_audio=False,  # 如果为True，Face Expert也能看到音频(解决下颚僵硬)，为False则彻底隔离
                 # [TimeFormer Control Switches]
                 use_tf_mouth=True,  # 嘴部是否应用 TimeFormer 偏移 (推荐 True)
                 use_tf_face=False  # 脸部是否应用 TimeFormer 偏移 (推荐 False，保持刚性)
                 ):
        super(MotionNetwork, self).__init__()

        # --- Audio Config ---
        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError

        self.bound = 0.15
        self.exp_eye = True
        self.individual_dim = ind_dim

        # --- Control Flags ---
        self.face_sees_audio = face_sees_audio
        self.use_tf_mouth = use_tf_mouth
        self.use_tf_face = use_tf_face
        print(f"\n[MotionNet] Initialized MoE:")
        print(f"  |-- Face sees Audio: {self.face_sees_audio}")
        print(f"  |-- TimeFormer on Mouth: {self.use_tf_mouth}")
        print(f"  |-- TimeFormer on Face:  {self.use_tf_face}")

        # --- Audio Encoder ---
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # --- Shared Spatial Encoder (HashGrid) ---
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        # 1. Mouth Encoders (只服务于嘴部专家)
        self.encoder_xy_mouth, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                            level_dim=self.level_dim, base_resolution=16,
                                                            log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz_mouth, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                            level_dim=self.level_dim, base_resolution=16,
                                                            log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz_mouth, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                            level_dim=self.level_dim, base_resolution=16,
                                                            log2_hashmap_size=17, desired_resolution=256 * self.bound)

        # 2. Face Encoders (只服务于面部专家)
        self.encoder_xy_face, _ = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                              level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17,
                                              desired_resolution=256 * self.bound)
        self.encoder_yz_face, _ = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                              level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17,
                                              desired_resolution=256 * self.bound)
        self.encoder_xz_face, _ = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                              level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17,
                                              desired_resolution=256 * self.bound)

        # --- Network Definitions ---
        self.num_layers = 3
        self.hidden_dim = 64
        #self.out_dim = 11  # rot: 4, xyz: 3, opac: 1, scale: 3
        # 定义输出维度
        self.out_dim_face = 11  # rot: 4, xyz: 3, opac: 1, scale: 3
        self.out_dim_mouth = 3  # !!! 修改点：Mouth 只输出 XYZ !!!
        # Expert A: Mouth Expert (Audio Driven)
        # Input: Spatial + Audio + Individual
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
        self.mouth_expert_mlp = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim_face, self.hidden_dim,
                                    self.num_layers)
        # 注意：这里输出维度改为 self.out_dim_mouth (3)
        self.mouth_expert_mlp = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim_mouth,
                                    self.hidden_dim, self.num_layers)
        # Expert B: Face Expert (Expression Driven)
        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)
        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # Calculate Face Input Dimension
        # Base: Spatial + Eye + Individual
        face_input_dim = self.in_dim + self.eye_dim + self.individual_dim
        if self.face_sees_audio:
            face_input_dim += self.audio_dim  # Add Audio dim if enabled

        self.face_expert_mlp = MLP(face_input_dim, self.out_dim_face, self.hidden_dim, self.num_layers)

        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1)

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz

    #def encode_x(self, xyz, bound):
        # xyz: [N, 3]
    #    xy, yz, xz = self.split_xyz(xyz)
    #    feat_xy = self.encoder_xy(xy, bound=bound)
    #    feat_yz = self.encoder_yz(yz, bound=bound)
    #    feat_xz = self.encoder_xz(xz, bound=bound)
    #    return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    # 两个独立的编码函数
    def encode_x_mouth(self, xyz, bound):
        xy, yz, xz = self.split_xyz(xyz)
        return torch.cat([
            self.encoder_xy_mouth(xy, bound=bound),
            self.encoder_yz_mouth(yz, bound=bound),
            self.encoder_xz_mouth(xz, bound=bound)
        ], dim=-1)

    def encode_x_face(self, xyz, bound):
        xy, yz, xz = self.split_xyz(xyz)
        return torch.cat([
            self.encoder_xy_face(xy, bound=bound),
            self.encoder_yz_face(yz, bound=bound),
            self.encoder_xz_face(xz, bound=bound)
        ], dim=-1)

    def encode_audio(self, a):
        if a is None: return None
        enc_a = self.audio_net(a)
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        return enc_a

    def forward(self, x, a, e=None, c=None, mouth_weight=None, time_offset=None):
        """
        Args:
            x: Canonical XYZ (Raw coordinates)
            a: Audio Features
            e: Expression Features
            c: Individual Code
            mouth_weight: [N, 1] Soft mask (1=Mouth, 0=Face)
            time_offset: [N, 3] TimeFormer predicted offsets
        """

        # ==========================================
        # 1. Prepare Coordinates for Experts
        # ==========================================
        x_mouth_in = x
        x_face_in = x

        if time_offset is not None:
            if self.use_tf_mouth:
                x_mouth_in = x + time_offset
            if self.use_tf_face:
                x_face_in = x + time_offset

        # 2. Spatial Encoding
        # Optimization: If both use the same coords (both True or both False), encode once

        #if self.use_tf_mouth == self.use_tf_face:
        #    enc_x_common = self.encode_x(x_mouth_in, bound=self.bound)
        #    enc_x_mouth = enc_x_common
        #    enc_x_face = enc_x_common
        #else:
        enc_x_mouth = self.encode_x_mouth(x_mouth_in, bound=self.bound)
        enc_x_face = self.encode_x_face(x_face_in, bound=self.bound)

        # Prepare Individual Code
        if c is not None:
            c = c.repeat(x.shape[0], 1)

        # Mouth Branch
        enc_a = self.encode_audio(a)
        enc_a_rep = enc_a.repeat(x.shape[0], 1)

        # Audio-Spatial Attention (Uses Mouth Coords)
        aud_ch_att = self.aud_ch_att_net(enc_x_mouth)
        enc_w = enc_a_rep * aud_ch_att

        if c is not None:
            h_mouth = torch.cat([enc_x_mouth, enc_w, c], dim=-1)
        else:
            h_mouth = torch.cat([enc_x_mouth, enc_w], dim=-1)

        #out_mouth = self.mouth_expert_mlp(h_mouth)
        # [CRITICAL CHANGE] Mouth 只输出 3 维 (XYZ)
        out_mouth = self.mouth_expert_mlp(h_mouth)  # [N, 3]

        # 4. Expert B: Face Branch
        if e is not None:
            # Eye Attention (Uses Face Coords)
            eye_att = torch.relu(self.eye_att_net(enc_x_face))
            enc_e = self.exp_encode_net(e[:-1])
            enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
            enc_e = enc_e * eye_att
        else:
            enc_e = torch.zeros((x.shape[0], self.eye_dim), device=x.device)
            eye_att = torch.zeros((x.shape[0], self.eye_dim), device=x.device)

        # Build Face Input Features
        face_feats = [enc_x_face, enc_e]

        # Debugging: Allow Face to see Audio?
        if self.face_sees_audio:
            # Re-use audio feature calculated above (minor approximation but efficient)
            face_feats.append(enc_w)

        if c is not None:
            face_feats.append(c)

        h_face = torch.cat(face_feats, dim=-1)
        out_face = self.face_expert_mlp(h_face)

        # ==========================================
        # 5. Output Processing & Safety Scaling
        # ==========================================

        # --- 处理 Face Expert 输出 ---
        # Face Expert 负责所有属性，作为 Base
        d_xyz_face = torch.tanh(out_face[..., :3]) * 1e-2
        d_rot_face = torch.tanh(out_face[..., 3:7]) * 0.2
        d_opa_face = out_face[..., 7:8]
        d_scale_face = torch.tanh(out_face[..., 8:11]) * 0.5

        # --- 处理 Mouth Expert 输出 ---
        # 1. 基础缩放 (防止初始值过大)
        d_xyz_mouth = torch.tanh(out_mouth)  * 1e-2

        # 2. [关键] 轴向抑制 (Axis Suppression)
        # 限制 X(左右) 和 Z(前后)，放开 Y(上下)
        # 这是一个 Trick：在 tanh 之后再除以一个系数，强行把这两个轴的位移压得非常小
        scale_vec = torch.tensor([0.2, 1.0, 0.2], device=x.device).unsqueeze(0)
        d_xyz_mouth = d_xyz_mouth * scale_vec
        '''
        # ==========================================
        # 5. MoE Fusion
        # ==========================================
        if mouth_weight is not None:
            if mouth_weight.dim() == 1:
                mouth_weight = mouth_weight.unsqueeze(1)
            # === [手动权重重映射逻辑] ===
            # 只要原始权重达到，就视为 100% 的嘴部区域
            threshold = 0.4

            # 线性放大权重：将 0~0.4 映射到 0~1.0
            # 这比 Sigmoid 更线性，梯度更平缓，不会炸
            gating_mask = torch.clamp(mouth_weight / threshold, min=0.0, max=1.0)

            # 平滑混合 (Linear Interpolation)
            final_out = gating_mask * out_mouth + (1.0 - gating_mask) * out_face
        else:
            # Fallback if no weight is provided (e.g. init phase)
            final_out = out_mouth + out_face
        '''
        # ==========================================
        # 6. MoE Fusion (Robust Version)
        # ==========================================
        if mouth_weight is not None:
            if mouth_weight.dim() == 1:
                mouth_weight = mouth_weight.unsqueeze(1)

            threshold = 0.4
            gating_mask = torch.clamp(mouth_weight / threshold, min=0.0, max=1.0)

            # [XYZ 融合] 只有 XYZ 进行混合
            final_d_xyz = gating_mask * d_xyz_mouth + (1.0 - gating_mask) * d_xyz_face

            # [其他属性] 强制使用 Face Expert 的结果
            # 因为 Mouth Expert 根本没有输出这些，且嘴部不应该独立旋转/缩放
            final_d_rot = d_rot_face
            final_d_scale = d_scale_face
            final_d_opa = d_opa_face

        else:
            # Fallback (通常只在第一帧或debug)
            final_d_xyz = d_xyz_mouth + d_xyz_face
            final_d_rot = d_rot_face
            final_d_scale = d_scale_face
            final_d_opa = d_opa_face



        return {
            'd_xyz': final_d_xyz,
            'd_rot': final_d_rot,
            'd_opa': final_d_opa,
            'd_scale': final_d_scale,
            # Return attention maps for visualization/loss
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True),
        }

    def get_params(self, lr, lr_net, wd=0):
        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            #{'params': self.encoder_xy.parameters(), 'lr': lr},
            #{'params': self.encoder_yz.parameters(), 'lr': lr},
            #{'params': self.encoder_xz.parameters(), 'lr': lr},

            # Mouth Encoders
            {'params': self.encoder_xy_mouth.parameters(), 'lr': lr},
            {'params': self.encoder_yz_mouth.parameters(), 'lr': lr},
            {'params': self.encoder_xz_mouth.parameters(), 'lr': lr},

            # Face Encoders
            {'params': self.encoder_xy_face.parameters(), 'lr': lr},
            {'params': self.encoder_yz_face.parameters(), 'lr': lr},
            {'params': self.encoder_xz_face.parameters(), 'lr': lr},

            # Experts
            {'params': self.mouth_expert_mlp.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.face_expert_mlp.parameters(), 'lr': lr_net, 'weight_decay': wd},

            # Attention & Helper Nets
            {'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})

        return params
