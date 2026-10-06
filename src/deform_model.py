import sys
import pickle
import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F
from pytorch3d.io import load_obj

from flame import FLAME_mica, parse_args
from utils.general_utils import Pytorch3dRasterizer, Embedder, load_binary_pickle, a_in_b_torch, face_vertices_gen


class Deform_Model(nn.Module):
    """
    形变模型：负责建立从FLAME的低维参数空间到高密度的3D高斯采样点云的映射。
    核心管线：规范空间UV采样 -> 动态网格蒙皮(LBS) -> 重心坐标插值 -> MLP残差补偿。
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        # 实例化FLAME核心模块。该模块提供基于PCA的几何形变（形状+表情）和基于LBS的姿态形变。
        mica_flame_config = parse_args()
        self.flame_model = FLAME_mica(mica_flame_config).to(self.device)

        # 定义规范空间(Canonical Space)的基准参数：形状零向量、表情零向量。
        self.default_shape_code = torch.zeros(1, 300, device=self.device)
        self.default_expr_code = torch.zeros(1, 100, device=self.device)
        
        # positional encoding
        # 高频位置编码(Positional Encoding)，用于将低维空间坐标映射到高维，
        # 以克服MLP对高频几何细节（如皱纹）的谱偏差(Spectral Bias)。
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)

        # 加载标准FLAME的UV拓扑。
        _, faces, aux = load_obj('flame/FlameMesh.obj', load_textures=False)
        uv_coords = aux.verts_uvs[None, ...]
        # 坐标系对齐：将 [0,1] 的UV坐标映射到 PyTorch3D 标准的 NDC (Normalized Device Coordinates) 空间 [-1, 1]。
        uv_coords = uv_coords * 2 - 1
        uv_coords[..., 1] = - uv_coords[..., 1]
        # 升维至齐次坐标 (x, y, 1.0) 以便输入光栅化管线
        self.uvcoords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0. + 1.], -1).to(self.device)
        self.uvfaces = faces.textures_idx[None, ...].to(self.device)
        self.tri_faces = faces.verts_idx[None, ...].to(self.device)
        
        # rasterizer
        # 初始化光栅化器，128x128的UV分辨率定义了后续高斯点的理论最大采样密度。
        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)
        
        # flame mask
        # 读取语义掩码，剥离计算冗余（如眼球后方、内耳等不可见区域的顶点）
        flame_mask_path = "flame/FLAME_masks/FLAME_masks.pkl"   #from MICA
        flame_mask_dic = load_binary_pickle(flame_mask_path)

        boundary_id = flame_mask_dic['boundary']
        # 提取脖子区域顶点索引。使用 .get() 方法提供空列表作为 fallback，提升应对非标资产文件的鲁棒性
        neck_id = flame_mask_dic.get('neck', [])

        # 将边界和脖子索引合并为统一的剔除集合
        exclude_ids = set(boundary_id).union(set(neck_id))

        full_id = np.array(range(5023)).astype(int)

        #neckhead_id_list = list(set(full_id) - set(boundary_id))
        # 集合相减：从全量顶点中同步剥离边界与脖子
        neckhead_id_list = list(set(full_id) - exclude_ids)

        self.neckhead_id_list = neckhead_id_list
        self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)
        self.init_networks()

    def init_networks(self):       
        ## full mica
        # MLP网络：负责预测超出FLAME网格表征能力的残差形变。
        # 输入维度：51(即3维位置经频率8编码后：3 + 3*2*8) + 120(表情100+下颌6+眼球12+眼睑2)。
        # 输出维度10：3(XYZ位移) + 4(旋转四元数) + 3(各向异性缩放)。
        self.deformNet = MLP(
            input_dim=self.pts_embedder.dim_embeded+120,
            output_dim=10,
            hidden_dim=256,
            hidden_layers=6
        )
        
    def example_init(self, codedict):
        """
        前置几何缓存计算：该步骤极为关键，计算特定身份(Shape)在规范空间下的连续表面到离散像素的固定映射(重心坐标)。
        此步骤的输出将在整个训练过程中被复用，以避免逐帧光栅化的高昂算力开销。
        """
        # speed up
        shape_code = codedict['shape'].detach().to(self.device)
        batch_size = shape_code.shape[0]
        # 解算处于零表情状态下的基础面部几何 (B, 5023, 3)
        geometry_shape = self.flame_model.forward_geo(
            shape_code,
            expression_params = self.default_expr_code
        )
        # 组织面索引数据进入光栅化器
        face_vertices_shape = face_vertices_gen(geometry_shape, self.tri_faces.expand(batch_size, -1, -1))
        # 执行硬光栅化(Hard Rasterization)
        # rast_out: (B, H, W, 4) 包含相交点三维坐标与掩码；pix_to_face: (B, H, W, 1) 相交面索引；bary_coords: 重心坐标
        rast_out, pix_to_face, bary_coords = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1),
                                         face_vertices_shape)
        self.pix_to_face_ori = pix_to_face
        self.bary_coords = bary_coords
        # 提取光栅化有效性掩码 (剔除未覆盖网格的背景UV像素)
        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1) # batch=1
        self.uvmask_flaten_idx = (uvmask_flaten[:]>0)
        # 压缩数据结构：仅保留有效像素对应的面索引和顶点索引
        pix_to_face_flaten = pix_to_face[0].clone().view(-1) # batch=1
        self.pix_to_face = pix_to_face_flaten[self.uvmask_flaten_idx] # pix to face idx
        self.pix_to_v_idx = self.tri_faces[0, self.pix_to_face, :] # pix to vert idx
        # 提取规范空间下的基准3D点云，并进行位置编码缓存
        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0) # batch=1       
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        self.uv_vertices_shape = uv_vertices_shape # for cano init
        self.uv_vertices_shape_embeded = self.pts_embedder(uv_vertices_shape)
        self.v_num = self.uv_vertices_shape_embeded.shape[1]    # 有效高斯基元总数

        # mask
        # 拓扑过滤：仅当相交三角面的三个顶点均位于有效头部/颈部掩码内时，才保留该采样点
        self.uv_head_idx = (
            a_in_b_torch(self.pix_to_v_idx[:,0], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,1], self.neckhead_id_tensor)
            & a_in_b_torch(self.pix_to_v_idx[:,2], self.neckhead_id_tensor)
        )
    
    def decode(self, codedict, keep_neck=False):
        """
        高频动态解算管线：依据输入条件实时驱动网格，并叠加MLP残差，输出高斯基元的空间物理状态。
        keep_neck=True 时保留脖子区域的高斯点。
        """
        shape_code = codedict['shape'].detach().to(self.device)
        expr_code = codedict['expr'].detach().to(self.device)
        jaw_pose = codedict['jaw_pose'].detach().to(self.device)
        eyelids = codedict['eyelids'].detach().to(self.device)
        eyes_pose = codedict['eyes_pose'].detach().to(self.device)
        batch_size = shape_code.shape[0]
        condition = torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1) # 拼接120维的条件控制向量

        # MLP
        # 张量扩张并拼接位置编码，送入MLP求解残差场
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        deforms = torch.tanh(deforms)   # 将网络输出钳制在[-1, 1]防止形变梯度爆炸

        # 拆解隐空间输出
        uv_vertices_deforms = deforms[..., :3]  # XYZ微观残差

        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)  # 对四元数实部求指数保证正定性
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)   # 四元数增量
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)  # 指数激活确保缩放系数域大于0
        # 执行FLAME前向LBS计算，获取当前帧的低频宏观变形网格
        geometry = self.flame_model.forward_geo(
            shape_code,
            expression_params=expr_code,
            jaw_pose_params=jaw_pose,
            eye_pose_params=eyes_pose,
            eyelid_params=eyelids,
        )
        face_vertices = face_vertices_gen(geometry, self.tri_faces.expand(batch_size, -1, -1))

        # rasterize face_vertices to uv space
        # 【核心操作】基于预缓存的光栅化结果，执行亚像素级几何映射
        D = face_vertices.shape[-1] # 3
        attributes = face_vertices.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        # 取出预存储的面索引
        idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        # 获取形变后网格面的三顶点当前坐标
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        # 使用预缓存的重心坐标在三顶点间执行严格插值，得出网格表面上连续的采样点精确坐标
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0) # batch=1
        uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        # 最终位置 = LBS基础形变插值坐标 + MLP高频微观残差
        verts_final = uv_vertices + uv_vertices_deforms

        # conduct mask
        # 应用掩码，剥除计算边界外无效数据
        if not keep_neck:
            verts_final = verts_final[:, self.uv_head_idx, :]
            rot_delta = rot_delta[:, self.uv_head_idx, :]
            scale_coef = scale_coef[:, self.uv_head_idx, :]

        return verts_final, rot_delta, scale_coef
    
    
    def capture(self):
        return (
            self.deformNet.state_dict(),
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args):
        (net_dict,
         opt_dict) = model_args
        self.deformNet.load_state_dict(net_dict)
        self.training_setup()
        self.optimizer.load_state_dict(opt_dict)

    
    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    
    def get_template(self):
        geometry_template = self.flame_model.forward_geo(
            self.default_shape_code,
            self.default_expr_code,
            self.default_jaw_pose,
            eye_pose_params=self.default_eyes_pose,
        )

        return geometry_template


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output
    
class MLP2(nn.Module):
    def __init__(self, input_dim, condition_dim, output_dim1, output_dim2, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.hidden_layers = hidden_layers
        mid_layers = math.ceil(hidden_layers/2.)
        self.input_dim = input_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        self.fcs1 = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i!=(mid_layers-1) else nn.Linear(hidden_dim, hidden_dim+output_dim1) for i in range(mid_layers)]
        )
        self.fcs2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) if i!=0 else nn.Linear(hidden_dim+condition_dim, hidden_dim) for i in range(mid_layers, hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim2)

    def forward(self, input, condition):
        # input: B,V,d1
        # condition: B,d2
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs1):
            h = self.fcs1[i](h)
            h = F.relu(h)
        oup1 = h[:, -self.output_dim1:]
        h = h[:, :-self.output_dim1]
        ...
        for i, l in enumerate(self.fcs2):
            h = self.fcs1[i](h)
            h = F.relu(h)
        # input_ori = input.reshape(batch_size*N_v, -1)
        # h = input_ori
        # for i, l in enumerate(self.fcs):
        #     h = self.fcs[i](h)
        #     h = F.relu(h)
        # output = self.output_linear(h)
        # output = output.reshape(batch_size, N_v, -1)


class SIRENMLP(nn.Module):
    def __init__(self,
                 input_dim=3,
                 output_dim=3,
                 hidden_dim=256,
                 hidden_layers=8,
                 condition_dim=100,
                 device=None):
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.z_dim = condition_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [FiLMLayer(self.input_dim, self.hidden_dim)] +
            [FiLMLayer(self.hidden_dim, self.hidden_dim) for i in range(self.hidden_layers - 1)]
        )
        self.final_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.mapping_network = MappingNetwork(condition_dim, 256,
                                              len(self.network) * self.hidden_dim * 2)

        self.network.apply(frequency_init(25))
        # self.final_layer.apply(frequency_init(25))
        self.final_layer.weight.data.normal_(0.0, 0.)
        self.final_layer.bias.data.fill_(0.)
        self.network[0].apply(first_layer_film_sine_init)

    def forward_vector(self, input, z):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts):
        frequencies = frequencies * 15 + 30
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)

        return sigma

    def forward(self, vertices, additional_conditioning):
        # vertices: in canonical space of flame
        # vertex eval: torch.Size([N, V, 3])
        # map eval:    torch.Size([N, 3, H, W])
        # conditioning
        # torch.Size([N, C])

        # vertex inputs (N, V, 3) -> (N, 3, V, 1)
        vertices = vertices.permute(0, 2, 1)[:, :, :, None]
        b, c, h, w = vertices.shape

        # to vector
        x = vertices.permute(0, 2, 3, 1).reshape(b, -1, c)

        z = additional_conditioning  # .repeat(1, h*w, 1)

        # apply siren network
        o = self.forward_vector(x, z)

        # to image
        o = o.reshape(b, h, w, self.output_dim)
        output = o.permute(0, 3, 1, 2)

        return output[:, :, :, 0].permute(0, 2, 1)

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift, ignore_conditions=None):
        x = self.layer(x)
        if ignore_conditions is not None:
            cond_freq, cond_phase_shift = freq[:-1], phase_shift[:-1]
            cond_freq = cond_freq.unsqueeze(1).expand_as(x).clone()
            cond_phase_shift = cond_phase_shift.unsqueeze(1).expand_as(x).clone()

            ignore_freq, ignore_phase_shift = freq[-1:], phase_shift[-1:]
            ignore_freq = ignore_freq.unsqueeze(1).expand_as(x)
            ignore_phase_shift = ignore_phase_shift.unsqueeze(1).expand_as(x)

            cond_freq[:, ignore_conditions] = ignore_freq[:, ignore_conditions]
            cond_phase_shift[:, ignore_conditions] = ignore_phase_shift[:, ignore_conditions]
            freq, phase_shift = cond_freq, cond_phase_shift

        else:
            freq = freq.unsqueeze(1).expand_as(x)
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)

        # print('x', x.shape)
        # print('freq', freq.shape)
        # print('phase_shift', phase_shift.shape)
        # x torch.Size([6, 5023, 256])
        # freq torch.Size([6, 5023, 256])
        # phase_shift torch.Size([6, 5023, 256])
        return torch.sin(freq * x + phase_shift)
    
def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
            elif isinstance(m, nn.Conv2d):
                num_input = torch.prod(
                    torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

    return init

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

        return frequencies, phase_shifts
    
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
        elif isinstance(m, nn.Conv2d):
            num_input = torch.prod(
                torch.tensor(m.weight.shape[1:], device=m.weight.device)).cpu().item()
            m.weight.uniform_(-1 / num_input, 1 / num_input)
