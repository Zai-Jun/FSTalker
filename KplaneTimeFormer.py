import itertools
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Sequence, Collection, Iterable, Optional
from utils.general_utils import get_expon_lr_func


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:

    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if grid_dim not in [2, 3]:
        raise NotImplementedError(f"Grid-sample implemented for 2D/3D only.")

    coords = coords.reshape([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))

    B, feature_dim = grid.shape[:2]
    B_coords = coords.shape[0]

    if B != B_coords:
        if B == 1:
            grid = grid.expand(B_coords, -1, -1, -1)
            B = B_coords
        else:
            raise RuntimeError(f"Batch mismatch: grid {B}, coords {B_coords}")

    n = coords.shape[-2]
    interp = F.grid_sample(
        grid, coords, align_corners=align_corners, mode='bilinear', padding_mode='border'
    )

    interp = interp.reshape(B, feature_dim, n).transpose(-1, -2).contiguous()
    return interp


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(grid_nd, in_dim, out_dim, reso, a=0.1, b=0.5):
    assert in_dim == len(reso)
    has_time_planes = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time_planes and 3 in coo_comb:
            nn.init.ones_(new_grid_coef)  # 时间平面初始化为 1
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)
    return grid_coefs


def interpolate_ms_features(pts, ms_grids, grid_dimensions, concat_features, num_levels):
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    if num_levels is None: num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.

    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]
            _interp = grid_sample_wrapper(grid[ci], pts[..., coo_comb])
            interp_out_plane = _interp.view(-1, feature_dim)
            interp_space = interp_space * interp_out_plane

        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class KPlaneEmbedder(nn.Module):
    def __init__(self, input_dims=4, output_dim=32, aabb=None, resolutions=[64, 128, 256, 512]):
        super().__init__()
        self.input_dim = input_dims
        self.output_dim = output_dim

        if aabb is None:
            aabb = torch.tensor([[-1.0] * input_dims, [1.0] * input_dims], dtype=torch.float32)
        self.register_buffer("aabb", aabb)

        num_levels = len(resolutions)
        features_per_level = output_dim // num_levels
        remainder = output_dim % num_levels

        self.grids = nn.ModuleList()
        for i, res in enumerate(resolutions):
            dim = features_per_level + (1 if i < remainder else 0)
            grid_res = [res] * input_dims
            gp = init_grid_param(grid_nd=2, in_dim=input_dims, out_dim=dim, reso=grid_res, a=0.1, b=0.5)
            self.grids.append(gp)

    def forward(self, inputs):
        original_shape = inputs.shape
        x_flat = inputs.reshape(-1, self.input_dim)

        x_norm = normalize_aabb(x_flat, self.aabb)
        features = interpolate_ms_features(x_norm, ms_grids=self.grids, grid_dimensions=2, concat_features=True,
                                           num_levels=len(self.grids))

        return features.reshape(*original_shape[:-1], self.output_dim)


def get_embedder(multires, input_dims=4):
    output_dim = 32
    base_res = 64
    resolutions = [base_res * (2 ** i) for i in range(multires)]
    embedder_obj = KPlaneEmbedder(input_dims=input_dims, output_dim=output_dim, resolutions=resolutions)
    return embedder_obj, output_dim


class TimeFormer(nn.Module):
    def __init__(self, training_args):
        super().__init__()

        # 1. 提取参数
        self.spatial_lr_scale = getattr(training_args, 'spatial_lr_scale', 5.0)  # 原版是 5
        self.position_lr_init = getattr(training_args, 'position_lr_init', 0.00016)
        self.position_lr_final = getattr(training_args, 'position_lr_final', 0.0000016)
        self.position_lr_delay_mult = getattr(training_args, 'position_lr_delay_mult', 0.02)
        self.deform_lr_max_steps = getattr(training_args, 'deform_lr_max_steps', 30000)

        input_dims = getattr(training_args, 'input_dims', 4)
        nhead = getattr(training_args, 'nhead', 4)
        dim_feedforward = getattr(training_args, 'dim_feedforward', 64)
        num_layers = getattr(training_args, 'num_layers', 1)
        multires = 4

        # 2. 初始化 K-Plane Embedder
        self.embed_fn, self.embed_ch = get_embedder(multires=multires, input_dims=input_dims)

        # 3. 初始化 Transformer Encoder
        # [CRITICAL Fix]: batch_first=False (默认值)
        # 这样输入 [4, N, C] 会被视为 Sequence=4, Batch=N
        # 实现了对每个点在时间轴上的 Attention，而不是点之间的 Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_ch,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=F.tanh,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出层
        self.output_layer = nn.Linear(self.embed_ch, 3)  # Output XYZ Offset
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def train_setting(self, training_args):

        self.setup_optimizer(training_args)

    def setup_optimizer(self, training_args):
        l = [
            {'params': list(self.transformer_encoder.parameters()),
             'lr': self.position_lr_init * self.spatial_lr_scale, "name": "deform_encoder"},
            {'params': list(self.output_layer.parameters()),
             'lr': self.position_lr_init * self.spatial_lr_scale, "name": "timeformer_decoder"},
            # 加入 K-Plane 参数
            {'params': list(self.embed_fn.parameters()),
             'lr': self.position_lr_init * self.spatial_lr_scale, "name": "kplane_embedder"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=self.position_lr_init * self.spatial_lr_scale,
            lr_final=self.position_lr_final,
            lr_delay_mult=self.position_lr_delay_mult,
            max_steps=self.deform_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.deform_scheduler_args(iteration)
            param_group['lr'] = lr
        return lr

    def forward(self, src):
        # src Shape: [Se
        # q=4, Batch=N, Dim=4]
        # 1. K-Plane 编码
        src_embed = self.embed_fn(src)  # -> [4, N, 32]

        # 2. Transformer (Seq=4, Batch=N)
        # 在 4 个时间步之间交换信息，内存占用为 O(4^2 * N)，完全可控
        h = self.transformer_encoder(src_embed)

        # 3. 输出
        h = self.output_layer(h)
        return h