#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from pytorch3d.transforms import matrix_to_rotation_6d
from torchvision.utils import save_image
import os
import random
import torch
import numpy as np

from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, normalize, patch_norm_mse_loss, \
    patch_norm_mse_loss_global, patch_norm_mse_loss_global2, patch_norm_mse_loss2
from gaussian_renderer import render, render_motion, render_dual
import sys
from scene import Scene, GaussianModel, MotionNetwork
from utils.general_utils import safe_state, get_expon_lr_func
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import loadCamOnTheFly
import copy
from matplotlib import pyplot as plt
import math
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from torch import nn

from src.deform_model import Deform_Model
from networks.dual_branch import DualBranchUNet
from utils.uv_utils import PositionMapGenerator, load_uv_region_masks
from utils.dual_utils import _build_uv_coords_from_grid, _compute_displacement, _to_axis_angle, _to_6d
from scene.dataset_readers import storePly
import torch.nn.functional as F


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _icp_align(source, target, max_iter=5):
    """ICP：将 source 点云刚性对齐到 target，返回 R[3,3], t[3]"""
    src = source.clone()
    for _ in range(max_iter):
        dists = torch.cdist(src, target)                 # [V, N]
        nn_idx = dists.argmin(dim=1)                      # [V]
        matched = target[nn_idx]                          # [V, 3]
        src_mean = src.mean(dim=0, keepdim=True)
        tgt_mean = matched.mean(dim=0, keepdim=True)
        H = (src - src_mean).T @ (matched - tgt_mean)    # [3, 3]
        U, S, V = torch.svd(H)
        R = V @ U.T
        detR = R[0,0]*(R[1,1]*R[2,2]-R[1,2]*R[2,1]) - R[0,1]*(R[1,0]*R[2,2]-R[1,2]*R[2,0]) + R[0,2]*(R[1,0]*R[2,1]-R[1,1]*R[2,0])
        if detR < 0:
            V[:, -1] *= -1
            R = V @ U.T
        t = (tgt_mean - src_mean @ R.T).squeeze(0)
        src = (R @ src.T).T + t
    return R, t




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    testing_iterations = [1] + [50] + [500] + [3000] + [i for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations = saving_iterations = [1] + [50] + [500] + [3000] + [i for i in
                                                                               range(0, opt.iterations + 1, 10000)] + [
                                                    opt.iterations]

    # ================= [Vars Setup] =================
    warm_step = 3000
    opt.densify_until_iter = opt.iterations - 1000
    bg_iter = opt.iterations  # opt.densify_until_iter
    lpips_start_iter = opt.densify_until_iter - 2000
    motion_stop_iter = bg_iter
    mouth_select_iter = bg_iter - 10000
    mouth_step = 1 / mouth_select_iter
    hair_mask_interval = 7
    select_interval = 15

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    ## deform model
    DeformModel = Deform_Model(device="cuda").to(device="cuda")
    DeformModel.training_setup()

    motion_net = MotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (
            0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / bg_iter))

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)

    dual_branch_net = None
    optimizer_dual = None
    map_gen = None
    align_R_tg = None   # 保存 iteration==1 的相机参数，供 warm_step ICP 使用
    align_T_tg = None
    _v_to_uv_cache = None  # ICP 顶点到 UV 映射缓存
    _flame_part_cache = None  # ICP 头部顶点语义分类缓存


    # ================= [CRITICAL FIX: 注入缺失参数] =================
    # TimeFormer 需要知道变形网络的学习率调度参数，但 TalkingGaussian 的参数列表里没有
    # 我们在这里手动补齐，数值参考常规 Deformable-3DGS 设置
    if not hasattr(opt, 'deform_lr_max_steps'):
        opt.deform_lr_max_steps = opt.iterations  # 通常设为总迭代次数
    # 参数补齐
    if not hasattr(opt, 'deform_lr_init'):
        opt.deform_lr_init = 0.00016
    if not hasattr(opt, 'deform_lr_final'):
        opt.deform_lr_final = 0.000016
    if not hasattr(opt, 'deform_lr_delay_mult'):
        opt.deform_lr_delay_mult = 0.01
    # ==============================================================

    # ======================================================

    bg_color = [0, 1, 0]  # [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ================= [1. Main Camera Selection (Curriculum)] =================
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # ================   FLAME初始 (全局仅执行一次)   =====
        if iteration == 1:
            with torch.no_grad():
                codedict = {}
                # 使用真实数据初始化
                B = 1
                td = viewpoint_cam.talking_dict
                target_device = "cuda"

                def _ensure_2d(t, dim):
                    if t is None:
                        return torch.zeros(B, dim, device=target_device)
                    t = t.float().to(target_device)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    return t

                codedict['shape'] = _ensure_2d(td.get('shape'), 300)
                codedict['expr'] = _ensure_2d(td.get('expr'), 100)
                codedict['eyelids'] = _ensure_2d(td.get('eyelids'), 2)

                I_mat = torch.eye(3, device=target_device, dtype=torch.float32).unsqueeze(0)
                I_6d = matrix_to_rotation_6d(I_mat)
                jaw_raw = _ensure_2d(td.get('jaw_pose'), 6)
                codedict['jaw_pose'] = _to_6d(jaw_raw) if td.get('jaw_pose') is not None else I_6d
                eyes_raw = _ensure_2d(td.get('eyes_pose'), 12)
                codedict['eyes_pose'] = _to_6d(eyes_raw, expected_6d_dim=12) if td.get('eyes_pose') is not None else torch.cat([I_6d, I_6d], dim=1)


                DeformModel.example_init(codedict)
                verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

            verts_flame = verts_final[0].detach()

            # ========================================================
            # 对齐（原版三锁）
            # ========================================================
            if hasattr(viewpoint_cam, 'width') and hasattr(viewpoint_cam, 'height'):
                W, H = int(viewpoint_cam.width), int(viewpoint_cam.height)
            else:
                with Image.open(viewpoint_cam.image_path) as _img:
                    W, H = _img.size

            fx_tg = W / (2 * math.tan(viewpoint_cam.FoVx / 2.0))
            fy_tg = H / (2 * math.tan(viewpoint_cam.FoVy / 2.0))
            cx_tg, cy_tg = W / 2.0, H / 2.0

            R_tg = torch.tensor(viewpoint_cam.R, device="cuda", dtype=torch.float32)
            T_tg = torch.tensor(viewpoint_cam.T, device="cuda", dtype=torch.float32)
            align_R_tg = R_tg
            align_T_tg = T_tg

            # 对齐：仅 Y/Z flip + depth offset
            verts_cam_tg = verts_flame.clone()
            verts_cam_tg[:, 1] *= -1
            verts_cam_tg[:, 2] *= -1
            verts_cam_tg[:, 2] += T_tg[2].item()
            # ========================================================

            init_verts_aligned = torch.matmul(verts_cam_tg - T_tg, torch.inverse(R_tg))
            gaussians.create_from_verts(init_verts_aligned)
            gaussians.training_setup(opt)

            # ============ DualBranchUNet 初始化 ============
            uv_size = DeformModel.uv_size
            canonical_verts = DeformModel.flame_model.forward_geo(
                shape_params=DeformModel.default_shape_code,
                expression_params=DeformModel.default_expr_code,
            )
            map_gen = PositionMapGenerator(
                verts=canonical_verts,
                faces=DeformModel.tri_faces,
                uvfaces=DeformModel.uvfaces,
                uv_coords=DeformModel.uvcoords[..., :2],
                image_size=uv_size,
                uv_size=uv_size,
                device="cuda",
            )
            position_map = map_gen.generate_position_map()

            uv_sample_coords = _build_uv_coords_from_grid(
                DeformModel.uvmask_flaten_idx,
                DeformModel.uv_head_idx,
                uv_size,
                "cuda",
            )
            uv_mask = load_uv_region_masks("flame/uv_region_masks.pkl")

            # 构建 UV head mask
            uv_head_mask = torch.zeros(1, 1, uv_size * uv_size, device="cuda")
            all_valid_idx = DeformModel.uvmask_flaten_idx.nonzero(as_tuple=False).squeeze(-1)
            head_in_valid = DeformModel.uv_head_idx.nonzero(as_tuple=False).squeeze(-1)
            uv_head_mask[:, :, all_valid_idx[head_in_valid]] = 1.0
            uv_head_mask = uv_head_mask.view(1, 1, uv_size, uv_size)

            gaussians._uv_coords = uv_sample_coords.squeeze(0)
            for k in uv_mask:
                m = uv_mask[k]
                if m.dim() == 3:
                    m = m.unsqueeze(1)
                if m.shape[-1] != uv_size:
                    m = F.interpolate(m.float(), size=(uv_size, uv_size),
                                      mode='bilinear', align_corners=True)
                uv_mask[k] = (m > 0.5).float()

            ref_img_path = os.path.join(dataset.source_path, 'gt_imgs', '0.jpg')
            reference_image = Image.open(ref_img_path).convert("RGB")

            dual_branch_net = DualBranchUNet(
                device="cuda",
                uv_sample_coords=uv_sample_coords,
                uv_mask=uv_mask,
                reference_image=reference_image,
                position_map=position_map,
                uv_size=uv_size,
                head_mask=uv_head_mask,
            ).to("cuda")
            dual_branch_net.train()
            optimizer_dual = torch.optim.Adam(dual_branch_net.parameters(), lr=opt.dual_branch_lr
                                               if hasattr(opt, 'dual_branch_lr') else 1e-4)
            print(f"[DualBranchUNet] Initialized, uv_size={uv_size}, uv_coords={uv_sample_coords.shape[1]}")
            # =============================================

        ###

        # find a big mouth strategy (Curriculum Learning)
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.2

        if iteration == 1:
            print(f"[Mouth] global lb={mouth_global_lb:.1f} ub={mouth_global_ub:.1f} range={mouth_global_ub-mouth_global_lb:.2f}")

        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        mouth_ub = mouth_lb + mouth_window
        mouth_lb = mouth_lb - mouth_window

        au_global_lb = 0
        au_global_ub = 1
        au_window = 0.3

        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)
        au_ub = au_lb + au_window
        au_lb = au_lb - au_window * 0.5

        if iteration < warm_step:
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or \
                        viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if warm_step < iteration < mouth_select_iter:
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if viewpoint_cam.original_image == None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))



        # Render Setup
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # ================= [Mask Prep for Primary Stream] =================
        # 这部分 Mask 变量对应主相机 viewpoint_cams[0]
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()

        face_mask = face_mask + mouth_mask
        head_mask = face_mask + hair_mask

        if iteration > lpips_start_iter:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

        hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0



        # Freezing logic
        if iteration > motion_stop_iter:
            for param in motion_net.parameters():
                param.requires_grad = False
        if iteration > bg_iter:
            gaussians._xyz.requires_grad = False
            gaussians._opacity.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False

        total_loss = 0.0

        # ================= [STREAM 1: Base Stream (Audio+Expr+UV Dual)] =================
        if iteration < warm_step:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        else:
            offset_dual = None
            if dual_branch_net is not None:
                td = viewpoint_cam.talking_dict
                codedict_frame = {
                    'shape': td.get('shape', DeformModel.default_shape_code),
                    'expr': td.get('expr', DeformModel.default_expr_code),
                }
                displacement_map = _compute_displacement(DeformModel, map_gen, codedict_frame)

                # ===== ICP 重新绑定 UV 坐标 (每 1000 步) =====
                if iteration >= warm_step and iteration % 1000 == 0 and dual_branch_net is not None:
                    with torch.no_grad():
                        # 当前帧 FLAME 几何体
                        def _e(t, dim):
                            if t is None: return torch.zeros(1, dim, device="cuda")
                            t = t.float().to("cuda")
                            return t.unsqueeze(0) if t.dim() == 1 else t
                        flame_geo = DeformModel.flame_model.forward_geo(
                            shape_params=_e(td.get('shape'), 300),
                            expression_params=_e(td.get('expr'), 100),
                            jaw_pose_params=_to_6d(_e(td.get('jaw_pose'), 6)) if td.get('jaw_pose') is not None else None,
                            eye_pose_params=_to_6d(_e(td.get('eyes_pose'), 12), 12) if td.get('eyes_pose') is not None else None,
                        )
                        # FLAME → world（同 iteration==1 对齐变换）
                        flame_cam = flame_geo[0].clone()
                        flame_cam[:, 1] *= -1
                        flame_cam[:, 2] *= -1
                        flame_cam[:, 2] += align_T_tg[2].item()
                        flame_world = torch.matmul(flame_cam - align_T_tg, torch.inverse(align_R_tg))

                        # ICP 用头部顶点（不含脖子）
                        head_ids = torch.tensor(DeformModel.neckhead_id_list, device="cuda")
                        flame_world_head = flame_world[head_ids]

                        # 顶点 → UV 映射（缓存）
                        if _v_to_uv_cache is None:
                            tri_f = DeformModel.tri_faces[0].long()
                            uv_f = DeformModel.uvfaces[0].long()
                            _v_to_uv_cache = -torch.ones(flame_world.shape[0], dtype=torch.long, device="cuda")
                            for j in range(3):
                                _v_to_uv_cache[tri_f[:, j]] = uv_f[:, j]
                        head_uv_idx = _v_to_uv_cache[head_ids]
                        uv_ndc = DeformModel.uvcoords[0, :, :2].clone()
                        uv_ndc[:, 1] = -uv_ndc[:, 1]
                        flame_uv_head = ((uv_ndc + 1) / 2)[head_uv_idx]

                        # 高斯当前 3D 位置
                        audio_feat = viewpoint_cam.talking_dict["auds"].cuda()
                        exp_feat = viewpoint_cam.talking_dict["au_exp"].cuda()
                        motion_preds = motion_net(gaussians.get_xyz, audio_feat, exp_feat, None)
                        gaussian_world = gaussians.get_xyz + motion_preds['d_xyz']

                        # ICP 对齐
                        R_icp, t_icp = _icp_align(flame_world_head, gaussian_world)
                        flame_aligned = (R_icp @ flame_world_head.T).T + t_icp

                        # 最近邻重新绑定 UV
                        dists = torch.cdist(gaussian_world, flame_aligned)
                        nn_idx = dists.argmin(dim=1)
                        new_uv = flame_uv_head[nn_idx]
                        gaussians._uv_coords = new_uv
                        dual_branch_net.update_uv_coords(new_uv.unsqueeze(0))
                        print(f"[ICP] iter={iteration}  head match={dists.min(dim=1).values.mean():.4f}  body max={dists.min(dim=1).values.max():.4f}")

                        # 生成头部语义映射点云图（不含脖子）
                        if _flame_part_cache is None:
                            from utils.general_utils import load_binary_pickle
                            masks = load_binary_pickle("flame/FLAME_masks/FLAME_masks.pkl")
                            _flame_part_cache = np.zeros(flame_world.shape[0], dtype=np.int8)
                            for region, pid in [('eye_region', 3), ('nose', 4), ('lips', 5), ('forehead', 6)]:
                                for v in masks.get(region, []):
                                    _flame_part_cache[v] = pid
                            for v in DeformModel.neckhead_id_list:
                                if _flame_part_cache[v] == 0:
                                    _flame_part_cache[v] = 1  # 头部其余部分
                        head_vertex_parts = _flame_part_cache[DeformModel.neckhead_id_list]
                        matched_part = head_vertex_parts[nn_idx.cpu().numpy()]
                        part_colors = {
                            0: [128,128,128], 1: [255,200,0], 3: [255,0,0],
                            4: [255,100,0], 5: [255,0,200], 6: [0,200,255],
                        }
                        part_rgb = np.array([part_colors.get(p, [128,128,128]) for p in matched_part], dtype=np.uint8)
                        parts_dir = os.path.join(dataset.model_path, "icp_parts")
                        os.makedirs(parts_dir, exist_ok=True)
                        storePly(os.path.join(parts_dir, f"iter_{iteration:05d}.ply"),
                                 gaussian_world.detach().cpu().numpy(), part_rgb)
                        part_rgb_01 = torch.from_numpy(part_rgb.astype(np.float32) / 255).to("cuda")
                        icp_screen = torch.zeros_like(gaussians.get_xyz, requires_grad=True, device="cuda") + 0
                        try: icp_screen.retain_grad()
                        except: pass
                        icp_rast = GaussianRasterizationSettings(
                            image_height=int(viewpoint_cam.image_height),
                            image_width=int(viewpoint_cam.image_width),
                            tanfovx=math.tan(viewpoint_cam.FoVx * 0.5),
                            tanfovy=math.tan(viewpoint_cam.FoVy * 0.5),
                            bg=torch.zeros(3, device="cuda"), scale_modifier=1.0,
                            viewmatrix=viewpoint_cam.world_view_transform,
                            projmatrix=viewpoint_cam.full_proj_transform,
                            sh_degree=0, campos=viewpoint_cam.camera_center,
                            prefiltered=False, debug=False)
                        icp_rast_out, _, _, _ = GaussianRasterizer(raster_settings=icp_rast)(
                            means3D=gaussian_world, means2D=icp_screen,
                            shs=None, colors_precomp=part_rgb_01,
                            opacities=gaussians.get_opacity,
                            scales=gaussians.get_scaling,
                            rotations=gaussians.get_rotation, cov3D_precomp=None)
                        parts_img = torch.clamp(icp_rast_out, 0, 1)
                        # GT + parts + overlay 三栏
                        gt_img = viewpoint_cam.original_image.cuda() / 255.0
                        overlay = gt_img * 0.5 + parts_img * 0.5
                        comparison = torch.cat([gt_img, parts_img, overlay], dim=2)
                        save_image(comparison, os.path.join(parts_dir, f"iter_{iteration:05d}.png"))

                flame_param = {
                    'expr': td.get('expr', torch.zeros(1, 100, device="cuda")).to("cuda"),
                    'neck_pose': _to_axis_angle(td.get('neck_pose', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                    'jaw_pose': _to_axis_angle(td.get('jaw_pose', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                    'eyes_pose': _to_axis_angle(td.get('eyes_pose', torch.zeros(1, 6, device="cuda")), 6).to("cuda"),
                    'rotation': _to_axis_angle(td.get('rotation', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                    'translation': td.get('translation', torch.zeros(1, 3, device="cuda")).to("cuda"),
                }
                old_cudnn = torch.backends.cudnn.enabled
                torch.backends.cudnn.enabled = False
                offset_dual = dual_branch_net(flame_param, 0, displacement_map)
                torch.backends.cudnn.enabled = old_cudnn

                if iteration == warm_step:
                    td = viewpoint_cam.talking_dict
                    print(f"[Dual] expr[:5]={td.get('expr', torch.zeros(100))[:5]}")
                    print(f"[Dual] jaw_pose={td.get('jaw_pose', torch.zeros(3))}")
                    with torch.no_grad():
                        d = displacement_map
                        print(f"[Dual] displacement_map min={d.min().item():.4f} max={d.max().item():.4f} nonzero={(d.abs()>1e-6).sum().item()}")
                if torch.isnan(offset_dual).any() or torch.isinf(offset_dual).any():
                    print(f"[DUAL NAN/INF iter={iteration}] nan={torch.isnan(offset_dual).sum()} inf={torch.isinf(offset_dual).sum()}")
                    for i, name in enumerate(['pos','scale','rot','color','opacity']):
                        ch = offset_dual[0, :, i*3:(i+1)*3] if i < 4 else offset_dual[0, :, 12:13]
                        if ch.numel() > 0:
                            print(f"  {name}: min={ch.min().item():.4f} max={ch.max().item():.4f}")

            if offset_dual is not None:
                render_pkg = render_dual(viewpoint_cam, gaussians, motion_net,
                                         offset_dual, pipe, background, return_attn=True)
            else:
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net,
                                           pipe, background, return_attn=True)

        image_white, alpha, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["alpha"], \
                render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda() / 255.0
        gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

        # Base Loss Calculation
        loss_base = 0.0
        if iteration < bg_iter:
            if hair_mask_iter:
                image_white[:, hair_mask] = background[:, None]
                gt_image_white[:, hair_mask] = background[:, None]

            Ll1 = l1_loss(image_white, gt_image_white)
            loss_base = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))

            if iteration > warm_step:
                loss_base += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
                loss_base += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
                loss_base += 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()
                loss_base += 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()

                if offset_dual is not None:
                    loss_base += opt.lambda_offset_scale * (torch.abs(offset_dual[0, :, 3:6] - 1).mean())
                    loss_base += opt.lambda_offset_color * (torch.abs(offset_dual[0, :, 10:13]).mean())

                loss_base += 1e-3 * (((1 - alpha) * head_mask).mean() + (alpha * ~head_mask).mean())

                [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                if xmax > xmin and ymax > ymin:
                    loss_base += 1e-4 * render_pkg["attn"][1, xmin:xmax, ymin:ymax].mean()
                if not hair_mask_iter:
                    a1 = render_pkg["attn"][1][hair_mask]
                    a0 = render_pkg["attn"][0][hair_mask]
                    if a1.numel() > 0:
                        loss_base += 1e-4 * a1.mean()
                    if a0.numel() > 0:
                        loss_base += 1e-4 * a0.mean()


            image_t = image_white.clone()
            gt_image_t = gt_image_white.clone()

        else:
            # with real bg
            image = image_white - background[:, None, None] * (
                    1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (
                            1.0 - alpha)
            Ll1 = l1_loss(image, gt_image)
            loss_base = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            image_t = image.clone()
            gt_image_t = gt_image.clone()

        if iteration > lpips_start_iter:
            [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
            if xmax > xmin and ymax > ymin:
                loss_base += 0.01 * lpips_criterion(image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1,
                                                    gt_image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1).mean()
            patch_size = random.randint(32, 48) * 2
            loss_base += 0.2 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size),
                                               patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()

        # 加入 Base Loss 到总 Loss
        total_loss += loss_base

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[LOSS NAN/INF iter={iteration}] total={total_loss.item():.4f} Ll1={Ll1.item():.4f}")
            if iteration > warm_step:
                img = render_pkg.get("render")
                if img is not None:
                    print(f"  render min={img.min().item():.4f} max={img.max().item():.4f} nan={torch.isnan(img).sum()} inf={torch.isinf(img).sum()}")
                mot = render_pkg.get("motion", {})
                for k in ['d_xyz','d_rot','d_opa','d_scale']:
                    if k in mot:
                        print(f"  motion.{k} nan={torch.isnan(mot[k]).sum()} inf={torch.isinf(mot[k]).sum()}")


        # Backward
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar (Log total_loss)
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{5}f}", "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss_base, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, motion_net,
                            render if iteration < warm_step else (render_dual if dual_branch_net is not
                                                                                 None else render_motion), (pipe, background),
                            dual_branch_net=dual_branch_net, deform_model=DeformModel,
                            map_gen=map_gen if dual_branch_net is not None else None)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration) + '_face')

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = [gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(),
                        timeFormer.state_dict(), timeFormer.optimizer.state_dict(),
                        iteration]
                if dual_branch_net is not None:
                    ckpt.append(dual_branch_net.state_dict())
                    ckpt.append(optimizer_dual.state_dict())
                    ckpt.append(gaussians._uv_coords)
                torch.save(ckpt, scene.model_path + "/chkpnt_face_" + str(iteration) + ".pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_face_latest" + ".pth")

            # Densification (只基于 Base Stream 的 viewspace_point_tensor)
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold,
                                                0.05 + 0.25 * iteration / opt.densify_until_iter,
                                                scene.cameras_extent, size_threshold)

            # bg prune
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh
                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                bg_color_mask = (colors_precomp[..., 0] < 30 / 255) * (colors_precomp[..., 1] > 225 / 255) * (
                        colors_precomp[..., 2] < 30 / 255)
                gaussians.prune_points(bg_color_mask.squeeze())

            # 在所有 pruning 完成后同步 uv_coords 到网络
            if dual_branch_net is not None:
                dual_branch_net.update_uv_coords(gaussians.get_uv_coords.unsqueeze(0))

            # Optimizer step
            if iteration < opt.iterations:
                motion_optimizer.step()
                gaussians.optimizer.step()
                timeFormer.optimizer.step()

                motion_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none=True)
                timeFormer.optimizer.zero_grad()

                if optimizer_dual is not None:
                    optimizer_dual.step()
                    optimizer_dual.zero_grad(set_to_none=True)

                scheduler.step()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # --- 修改开始：强制禁用 TensorBoard ---
    print("[INFO] TensorBoard disabled to prevent protobuf crash.")
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # --- 修改结束 ---
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, motion_net,
                    renderFunc, renderArgs, dual_branch_net=None, deform_model=None, map_gen=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test',
                               'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in
                                           range(5, 100, 5)]},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if viewpoint.original_image == None:
                        viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))

                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    elif renderFunc is render_dual and dual_branch_net is not None:
                        td = viewpoint.talking_dict
                        codedict_frame = {'shape': td.get('shape', deform_model.default_shape_code),
                                          'expr': td.get('expr', deform_model.default_expr_code)}
                        displacement_map = _compute_displacement(deform_model, map_gen, codedict_frame)
                        if idx == 0:
                            print(f"[UV diag {config['name']}] expr in td={'expr' in td}, jaw={'jaw_pose' in td}, shape={td.get('shape', '?').shape if 'shape' in td else 'MISSING'}")
                            print(f"[UV diag {config['name']}] displacement nonzero={(displacement_map.abs()>1e-7).sum().item()} min={displacement_map.min().item():.5f} max={displacement_map.max().item():.5f}")
                        flame_param = {
                            'expr': td.get('expr', torch.zeros(1, 100, device="cuda")).to("cuda"),
                            'neck_pose': _to_axis_angle(td.get('neck_pose', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                            'jaw_pose': _to_axis_angle(td.get('jaw_pose', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                            'eyes_pose': _to_axis_angle(td.get('eyes_pose', torch.zeros(1, 6, device="cuda")), 6).to("cuda"),
                            'rotation': _to_axis_angle(td.get('rotation', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                            'translation': td.get('translation', torch.zeros(1, 3, device="cuda")).to("cuda"),
                        }
                        # 获取 raw offset_map 用于可视化
                        if idx < 5:
                            offset_dual, offset_map = dual_branch_net(flame_param, 0, displacement_map, return_map=True)
                        else:
                            offset_dual = dual_branch_net(flame_param, 0, displacement_map)
                            offset_map = None
                        render_pkg = render_dual(viewpoint, scene.gaussians, motion_net,
                                                 offset_dual, renderArgs[0], renderArgs[1], return_attn=True)
                        # 渲染无双分支版本用于对比
                        render_no_dual = render_motion(viewpoint, scene.gaussians, motion_net,
                                                       renderArgs[0], renderArgs[1], return_attn=False)
                    else:
                        render_pkg = render_motion(viewpoint, scene.gaussians, motion_net,
                                                   return_attn=True, frame_idx=0, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    image = image - renderArgs[1][:, None, None] * (
                                1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)

                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                             gt_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name),
                                             (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name),
                                             (~mouth_mask[None] * gt_image)[None], global_step=iteration)

                        if renderFunc is not render:
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name),
                                                 (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None],
                                                 global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name),
                                                 (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None],
                                                 global_step=iteration)
                    # ================= [可视化：四栏对比 + UV 热力图] =================
                    if idx < 5:
                        vis_dir = os.path.join(scene.model_path, "visualizations", config['name'])
                        os.makedirs(vis_dir, exist_ok=True)

                        # 四栏对比图：GT | motion_only | dual | diff×5
                        if renderFunc is render_dual and offset_map is not None:
                            image_no_dual = torch.clamp(render_no_dual["render"], 0.0, 1.0)
                            image_no_dual = image_no_dual - renderArgs[1][:, None, None] * (
                                1.0 - render_no_dual["alpha"]) + viewpoint.background.cuda() / 255.0 * (
                                1.0 - render_no_dual["alpha"])
                            diff_img = (image - image_no_dual).abs() * 5.0
                            diff_img = torch.clamp(diff_img, 0.0, 1.0)
                            comparison_img = torch.cat([gt_image, image_no_dual, image, diff_img], dim=2)
                            save_path = os.path.join(vis_dir, f"iter_{iteration:05d}_view_{viewpoint.image_name}.png")
                            save_image(comparison_img, save_path)

                            # UV 位移热力图
                            heat = torch.norm(offset_map[0, :3], dim=0)  # [H,W]
                            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                            heat_cmap = torch.cat([
                                heat.unsqueeze(0),
                                torch.zeros_like(heat).unsqueeze(0),
                                1.0 - heat.unsqueeze(0),
                            ], dim=0)
                            save_image(heat_cmap, os.path.join(vis_dir, f"iter_{iteration:05d}_view_{viewpoint.image_name}_uvheat.png"))

                            # 点云位移图：各高斯点的位移量级染色
                            motion_xyz = scene.gaussians.get_xyz + render_no_dual["motion"]['d_xyz']
                            final_xyz = motion_xyz + offset_dual[0, :, :3]
                            disp_mag = torch.norm(offset_dual[0, :, :3], dim=1)
                            disp_norm = (disp_mag - disp_mag.min()) / (disp_mag.max() - disp_mag.min() + 1e-8)
                            r = torch.clamp(disp_norm * 2 - 0.5, 0, 1)
                            g = torch.clamp(disp_norm * 1.5 - 0.25, 0, 1) * 0.8
                            b = torch.clamp(1.5 - disp_norm * 2, 0, 1)
                            rgb = (torch.stack([r, g, b], dim=1) * 255).clamp(0, 255).byte().cpu().numpy()
                            storePly(os.path.join(vis_dir, f"iter_{iteration:05d}_view_{viewpoint.image_name}_disp.ply"),
                                     final_xyz.detach().cpu().numpy(), rgb)

                            # 将位移点云渲染为图片
                            disp_rgb_01 = torch.stack([r, g, b], dim=1)  # [N,3] in [0,1]
                            disp_screenspace = torch.zeros_like(scene.gaussians.get_xyz)
                            disp_raster_settings = GaussianRasterizationSettings(
                                image_height=int(viewpoint.image_height),
                                image_width=int(viewpoint.image_width),
                                tanfovx=math.tan(viewpoint.FoVx * 0.5),
                                tanfovy=math.tan(viewpoint.FoVy * 0.5),
                                bg=torch.zeros(3, device="cuda"),
                                scale_modifier=1.0,
                                viewmatrix=viewpoint.world_view_transform,
                                projmatrix=viewpoint.full_proj_transform,
                                sh_degree=0,
                                campos=viewpoint.camera_center,
                                prefiltered=False, debug=False,
                            )
                            disp_rasterizer = GaussianRasterizer(raster_settings=disp_raster_settings)
                            render_disp, _, _, _ = disp_rasterizer(
                                means3D=final_xyz,
                                means2D=disp_screenspace,
                                shs=None,
                                colors_precomp=disp_rgb_01,
                                opacities=scene.gaussians.get_opacity,
                                scales=scene.gaussians.get_scaling,
                                rotations=scene.gaussians.get_rotation,
                                cov3D_precomp=None)
                            render_disp = torch.clamp(render_disp, 0.0, 1.0)
                            save_image(render_disp, os.path.join(vis_dir,
                                       f"iter_{iteration:05d}_view_{viewpoint.image_name}_disprender.png"))

                            # uvdisp：displacement_map 在每个高斯 UV 处的采样值 → 点云 + 渲染图
                            uv_grid = scene.gaussians.get_uv_coords * 2 - 1          # [N,2] [0,1]→[-1,1]
                            uv_grid = uv_grid.unsqueeze(0).unsqueeze(2)         # [1,N,1,2]
                            uv_sampled = F.grid_sample(displacement_map, uv_grid,
                                                       mode='bilinear', padding_mode='border', align_corners=True)
                            uv_sampled = uv_sampled.squeeze(3).squeeze(0).T    # [N,3]
                            uvdisp_mag = torch.norm(uv_sampled, dim=1)          # [N]
                            uvdisp_norm = (uvdisp_mag - uvdisp_mag.min()) / (uvdisp_mag.max() - uvdisp_mag.min() + 1e-8)
                            r2 = torch.clamp(uvdisp_norm * 2 - 0.5, 0, 1)
                            g2 = torch.clamp(uvdisp_norm * 1.5 - 0.25, 0, 1) * 0.8
                            b2 = torch.clamp(1.5 - uvdisp_norm * 2, 0, 1)
                            uv_rgb = (torch.stack([r2, g2, b2], dim=1) * 255).clamp(0, 255).byte().cpu().numpy()
                            storePly(os.path.join(vis_dir, f"iter_{iteration:05d}_view_{viewpoint.image_name}_uvdisp.ply"),
                                     motion_xyz.detach().cpu().numpy(), uv_rgb)
                            uv_rgb_01 = torch.stack([r2, g2, b2], dim=1)
                            render_uvdisp, _, _, _ = disp_rasterizer(
                                means3D=motion_xyz,
                                means2D=disp_screenspace,
                                shs=None, colors_precomp=uv_rgb_01,
                                opacities=scene.gaussians.get_opacity,
                                scales=scene.gaussians.get_scaling,
                                rotations=scene.gaussians.get_rotation, cov3D_precomp=None)
                            save_image(torch.clamp(render_uvdisp, 0, 1), os.path.join(vis_dir,
                                       f"iter_{iteration:05d}_view_{viewpoint.image_name}_uvdisp.png"))
                        else:
                            overlay_image = gt_image * 0.3 + image * 0.7
                            comparison_img = torch.cat([gt_image, image, overlay_image], dim=2)
                            save_path = os.path.join(vis_dir, f"iter_{iteration:05d}_view_{viewpoint.image_name}.png")
                            save_image(comparison_img, save_path)
                    # ==========================================================

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
