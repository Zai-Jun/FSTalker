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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork
from utils.sh_utils import eval_sh
from utils.general_utils import quatProduct_batch

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii}


def render_motion(viewpoint_camera, pc : GaussianModel, motion_net : MotionNetwork, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, frame_idx = None, return_attn = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()

    # ind_code = motion_net.individual_codes[frame_idx if frame_idx is not None else viewpoint_camera.talking_dict["img_id"]]
    ind_code = None
    motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, ind_code) #  
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    # opacity = pc.opacity_activation(pc._opacity + motion_preds['d_opa'])
    opacity = pc.get_opacity

    cov3D_precomp = None
    # scales = pc.get_scaling
    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])

    colors_precomp = None
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Attn
    rendered_attn = None
    if return_attn:
        attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'], torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
        rendered_attn, _, _, _ = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = attn_precomp,
            opacities = opacity.detach(),
            scales = scales.detach(),
            rotations = rotations.detach(),
            cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds,
            'attn': rendered_attn}

def render_dual(viewpoint_camera, pc, motion_net, offset_dual, pipe, bg_color, scaling_modifier=1.0, return_attn=False):
    """
    MotionNetwork + DualBranchUNet 并行渲染。
    MotionNetwork 负责低频形变 (d_xyz, d_rot, d_scale)，
    DualBranchUNet 的 13 维 offset 在此基础上叠加微调 (pos/scale/rot/color/opacity)。
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()
    ind_code = None

    motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, ind_code)

    offset_dual = offset_dual.squeeze(0)

    means3D = pc.get_xyz + motion_preds['d_xyz'] + offset_dual[:, :3]
    means2D = screenspace_points

    scales_base = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    scales = scales_base * offset_dual[:, 3:6]

    rots_base = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])
    rotations = quatProduct_batch(offset_dual[:, 6:10], rots_base)

    opacity = torch.clamp(pc.get_opacity + offset_dual[:, 12:13], 0, 1)

    cov3D_precomp = None
    shs = pc.get_features.clone()
    shs[:, 0, :3] = shs[:, 0, :3] + offset_dual[:, 9:12]
    colors_precomp = None

    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    rendered_attn = None
    if return_attn:
        attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'],
                                   torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
        rendered_attn, _, _, _ = rasterizer(
            means3D=means3D.detach(),
            means2D=means2D,
            shs=None,
            colors_precomp=attn_precomp,
            opacities=opacity.detach(),
            scales=scales.detach(),
            rotations=rotations.detach(),
            cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds,
            "attn": rendered_attn}


def render_timeformer(viewpoint_camera, pc, motion_net, time_offset, pipe, bg_color, scaling_modifier=1.0):
    """
    自定义渲染函数，支持传入 TimeFormer 的 offset。
    逻辑复刻自 gaussian_renderer/__init__.py 中的 render_motion
    """
    # 1. 准备光栅化设置 (与 render_motion 保持一致)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 2. 获取 TalkingGaussian 特有的驱动特征
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()
    ind_code = None  # 这里暂时保持为 None，如需支持 id 可从 talking_dict["img_id"] 获取

    # 3. TimeFormer 核心逻辑：应用时间偏移
    # 原始坐标 + TimeFormer Offset
    shifted_xyz = pc.get_xyz + time_offset

    # 4. 传入 MotionNet (注意：使用 shifted_xyz)
    # MotionNet forward: x, a, e=None, c=None
    motion_preds = motion_net(shifted_xyz, audio_feat, exp_feat, ind_code)

    # 5. 计算最终变形 (Base Gaussian + TimeFormer + MotionNet)
    means3D = shifted_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])

    shs = pc.get_features
    colors_precomp = None
    cov3D_precomp = None

    # 6. 光栅化
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds}


def render_moe(viewpoint_camera, pc, motion_net, pipe, bg_color, scaling_modifier=1.0, override_color=None,
               time_offset=None, return_attn=False, active_region=None):
    """
    [终极融合版] 统一的 MoE 渲染接口
    既支持 MoE/TimeFormer，也支持原版 Attention Loss 计算。
    """
    # 1. 准备光栅化设置
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 2. 准备网络输入
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()

    # [MoE 核心] 获取 mouth_weight
    # 注意：如果刚开始训练，weight 可能是 None 或 全0，MotionNet 内部要处理好
    mouth_weight = pc.get_mouth_weight

    # 3. TimeFormer 逻辑 (Offset 叠加)
    current_xyz = pc.get_xyz
    if time_offset is not None:
        current_xyz = current_xyz + time_offset

    # 4. MotionNet 前向传播 (传入 mouth_weight)
    ind_code = None
    motion_preds = motion_net(current_xyz, audio_feat, exp_feat, ind_code, mouth_weight=mouth_weight)

    # 5. 应用变形
    means3D = current_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    opacity = pc.get_opacity + motion_preds['d_opa']
    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])
    shs = pc.get_features

    # ================= [新增: 动态高斯剔除 (Dynamic Pruning)] =================
    # 在进入光栅化之前，把不需要的点扔掉！
    keep_mask = None

    if active_region is not None and mouth_weight is not None:
        # 确保 weight 维度正确
        w = mouth_weight.squeeze() if mouth_weight.dim() > 1 else mouth_weight

        if active_region == 'mouth':
            # 策略: 只保留有一点点嘴部权重的点 (阈值可调，比如 0.05)
            # 这样背景和头发(w=0)会被直接剔除，不参与渲染
            keep_mask = w > 0.05

        elif active_region == 'face':
            # 策略: 只保留嘴部权重低的点
            keep_mask = w < 0.95

    # 如果生成了 mask，执行剔除
    if keep_mask is not None:
        means3D = means3D[keep_mask]
        opacity = opacity[keep_mask]
        scales = scales[keep_mask]
        rotations = rotations[keep_mask]
        shs = shs[keep_mask]
        # 注意: screenspace_points 也需要切片，否则梯度回传会错位
        means2D = screenspace_points[keep_mask]
    else:
        means2D = screenspace_points
    # ======================================================================

    # 6. 主渲染 (Render Image)
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=override_color,
        opacities=opacity,
        scales=scales,
        rotations=rotations)

    # 7. [关键复刻] Attention 渲染 (用于 Loss)
    rendered_attn = None
    if return_attn:
        # 将 Audio Attention 和 Eye Attention 拼在一起渲染
        # RGB 通道: R=AudioAttn, G=EyeAttn, B=0
        attn_precomp = torch.cat([
            motion_preds['ambient_aud'],
            motion_preds['ambient_eye'],
            torch.zeros_like(motion_preds['ambient_eye'])
        ], dim=-1)

        # 再次光栅化，这次只渲染 Attention 颜色
        rendered_attn, _, _, _ = rasterizer(
            means3D=means3D.detach(),  # 梯度阻断，只算颜色梯度
            means2D=means2D,
            shs=None,
            colors_precomp=attn_precomp,
            opacities=opacity.detach(),
            scales=scales.detach(),
            rotations=rotations.detach())

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "radii": radii,
        "motion": motion_preds,
        "attn": rendered_attn  # 确保这里有值，否则 loss_base 报错
    }


