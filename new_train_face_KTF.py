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
import numpy as np
from PIL import Image
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import torch
from random import randint

from matplotlib import pyplot as plt

from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, normalize, patch_norm_mse_loss, \
    patch_norm_mse_loss_global, patch_norm_mse_loss_global2, patch_norm_mse_loss2
from gaussian_renderer import render, render_motion, render_motion_opa, render_motion_depth
import sys
from scene import Scene, GaussianModel, MotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.normal_utils import depth_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch
torch.autograd.set_detect_anomaly(True)

# [添加到头部 Imports 区域]
import math
from torch import nn
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import get_expon_lr_func
from KplaneTimeFormer import TimeFormer  # 确保目录下有 TimeFormer.py
import copy

# [添加到 training 函数定义之前]
def render_timeformer(viewpoint_camera, pc, motion_net, time_offset, pipe, bg_color, scaling_modifier=1.0):
    """
    自定义渲染管线：支持 TimeFormer 的 Offset 输入
    流程: (XYZ + Offset) -> MotionNet -> Render
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

    # 获取驱动特征
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()
    ind_code = None

    # === [关键点] ===
    # 1. 应用 TimeFormer 修正
    shifted_xyz = pc.get_xyz + time_offset

    # 2. 传入 MotionNet (使用修正后的坐标)
    motion_preds = motion_net(shifted_xyz, audio_feat, exp_feat, ind_code)

    # 3. 叠加最终变形
    means3D = shifted_xyz + motion_preds['d_xyz']

    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])
    shs = pc.get_features

    rendered_image, radii, rendered_depth, rendered_alpha,_,_,_ = rasterizer(
        means3D=means3D, means2D=means2D, shs=shs, colors_precomp=None,
        opacities=opacity, scales=scales, rotations=rotations)

    return {"render": rendered_image, "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0, "alpha": rendered_alpha, "motion": motion_preds}

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mode_long, pretrain_ckpt_path):
    testing_iterations = [1] + [i for i in range(0, opt.iterations + 1, 2000)] # 设置测试迭代点：从第1次开始，每10000次迭代测试一次
    # 设置检查点和保存点：每10000次迭代保存，加上最终迭代
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 2000)] + [opt.iterations]

    # vars
    # 训练阶段控制变量
    warm_step = 3000  # 预热阶段结束点
    opt.densify_until_iter = opt.iterations - 3000  # 高斯密度化停止点
    bg_iter = opt.iterations  # 背景切换点
    lpips_start_iter = opt.densify_until_iter - 1500  # LPIPS损失启动点
    motion_stop_iter = bg_iter  # 运动网络训练停止点
    mouth_select_iter = opt.iterations  # 嘴部运动选择结束点
    mouth_step = 1 / max(mouth_select_iter, 1)  # 嘴部运动采样步长
    hair_mask_interval = 7  # 头发掩码更新间隔
    select_interval = 8  # 嘴部运动采样间隔

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if tb_writer:
        print("True")
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)

    # 创建运动网络
    motion_net = MotionNetwork(args=dataset).cuda()

    # ================= [TimeFormer Init] =================
    batch_timeformer = 4  # Batch大小，显存不够改2
    weight_t_loss = 0.8  # 辅助分支权重
    weight_tf_reg = 0.0001  # 正则化权重

    # 补全参数 (防止报错)
    if not hasattr(opt, 'deform_lr_max_steps'): opt.deform_lr_max_steps = opt.iterations
    if not hasattr(opt, 'deform_lr_init'): opt.deform_lr_init = 0.00016
    if not hasattr(opt, 'deform_lr_final'): opt.deform_lr_final = 0.000016
    if not hasattr(opt, 'deform_lr_delay_mult'): opt.deform_lr_delay_mult = 0.01

    # 新增 K-Planes 和 Transformer 所需参数
    opt.input_dims = 4  # 输入维度 (XYZT)
    opt.output_dims = 3  # 输出维度 (XYZ Offset)
    opt.nhead = 4  # Transformer 头数
    opt.dim_feedforward = 64  # Transformer 隐藏层维度 (对应旧代码 hidden_dims)
    opt.num_layers = 1  # Transformer 层数
    opt.dropout = 0.0  # Dropout (可选)

    # 空间学习率缩放，如果原 args 没有定义，给一个默认值
    if not hasattr(opt, 'spatial_lr_scale'): opt.spatial_lr_scale = 1.0
    # 位置编码学习率，KPlaneEmbedder 将使用此学习率
    if not hasattr(opt, 'position_lr_init'): opt.position_lr_init = 0.00016
    if not hasattr(opt, 'position_lr_final'): opt.position_lr_final = 0.0000016
    if not hasattr(opt, 'position_lr_delay_mult'): opt.position_lr_delay_mult = 0.02

    # [修改步骤 2]: 初始化 TimeFormer
    # 之前是传递 kwargs，现在直接传递配置对象 opt
    try:
        timeFormer = TimeFormer(opt).cuda()
    except Exception as e:
        print(f"[Error] TimeFormer init failed: {e}")
        raise e

    timeFormer.setup_optimizer(opt)
    total_frames = len(scene.getTrainCameras())
    # ======================================================
    # 配置优化器（分层学习率）
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.5 ** (iter / opt.iterations))
    if mode_long:   # 长训练模式调整
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.1 ** (iter / opt.iterations))

    # Load pre-trained
    # 加载预训练运动网络
    if pretrain_ckpt_path is not None and os.path.exists(pretrain_ckpt_path):
        print(f"[Info] Loading pre-trained model from: {pretrain_ckpt_path}")
        (motion_params, _, _) = torch.load(pretrain_ckpt_path)
        motion_net.load_state_dict(motion_params)
    else:
        print("[Info] No pre-trained model found. Training from scratch.")

    # (model_params, _, _, _) = torch.load(os.path.join("output/pretrain4/macron/chkpnt_face_latest.pth"))
    # gaussians.neural_motion_grid.load_state_dict(model_params[-1])

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()    # 初始化LPIPS感知损失

    gaussians.training_setup(opt)   # 设置高斯模型优化器
    if checkpoint:  # 恢复训练检查点
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)

    if not mode_long:    # 短训练模式配置
        gaussians.max_sh_degree = 1 # 限制球谐函数阶数

    # 绿色背景配置（用于抠像）
    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 训练计时器
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 训练进度监控
    viewpoint_stack = None  # 视角数据栈
    ema_loss_for_log = 0.0  # 指数移动平均损失
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1  # 起始迭代号调整

    # 在训练循环之前初始化损失记录
    loss_records = {
        'total': [],
        'l1': [],
        'dssim': [],
        'normal_consistency': [],
        'depth_consistency': [],
        'motion_d_xyz': [],
        'motion_d_rot': [],
        'motion_d_opa': [],
        'motion_d_scale': [],
        'motion_p_xyz': [],
        'opacity_reg': [],
        'attn_lips': [],
        'attn_audio': [],
        'attn_expression': [],
        'lpips': [],
        'iterations': []
    }

    for iteration in range(first_iter, opt.iterations + 1):         # 主训练循环开始
        iter_start.record() # 记录迭代开始时间点（CUDA事件）
        # 更新高斯模型的学习率（自适应衰减策略）
        gaussians.update_learning_rate(iteration)
        timeFormer.update_learning_rate(iteration)  # 新增

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 每1000次迭代增加球谐函数阶数（提升细节表现能力）
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()   # 提升球谐函数阶数

        # Pick a random Camera
        # 获取训练视角
        if not viewpoint_stack: # 检查视角栈是否为空
            viewpoint_stack = scene.getTrainCameras().copy()    # 从场景复制训练视角
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 随机选择一个视角

        # find a big mouth
        # 嘴部动作范围计算
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]  # 下界
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]  # 上界
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2  # 调整下界
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.5  # 采样窗口大小

        # 动态嘴部采样范围
        # mouth_step * iteration实现随训练推进扩大的采样范围
        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        mouth_ub = mouth_lb + mouth_window
        mouth_lb = mouth_lb - mouth_window  # 扩展下界

        # 眨眼动作(AU)范围设置
        au_global_lb = 0  # 下界
        au_global_ub = 1  # 上界
        au_window = 0.4  # 采样窗口

        # 动态眨眼采样范围
        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)
        au_ub = au_lb + au_window
        au_lb = au_lb - au_window * 1.5  # 扩展下界

        if iteration < warm_step and iteration < mouth_select_iter:
            if iteration % select_interval == 0:
                # 筛选嘴部开口度在目标范围内的视角
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        if warm_step < iteration < mouth_select_iter:
            # 筛选眨眼程度在目标范围内的视角
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # ================= [Batch Construction] =================
        # 1. 主视角 (Main View)
        viewpoint_cams = [viewpoint_cam]

        # 2. 辅助视角 (Aux Views) - 仅在 Warmup 后启用
        if iteration > warm_step:
            all_cams = scene.getTrainCameras()  # 获取所有训练相机
            while len(viewpoint_cams) < batch_timeformer:
                # 随机采样一个辅助相机 (无需 loadCamOnTheFly，直接取)
                aux_cam = all_cams[randint(0, len(all_cams) - 1)]
                viewpoint_cams.append(aux_cam)
        # ========================================================

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True   # 启用渲染管线调试
        # 加载面部/头发/嘴部掩码
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        face_mask = face_mask + mouth_mask #
        head_mask = face_mask + hair_mask
        FACE_STAGE_ITER = opt.iterations - 2000

        if iteration <= FACE_STAGE_ITER:

        # 嘴部掩码精细化处理
            if iteration <= lpips_start_iter:
                max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                # 形态学闭运算：先膨胀后腐蚀，平滑边界
                mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()


            # 头发掩码更新控制
            hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0

            # 预热阶段渲染
            if iteration < warm_step:
                # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                enable_align = iteration > 1000 # 1000次迭代后启用对齐
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=enable_align)
                # for param in motion_net.parameters():
                #     param.requires_grad = False
            # 主训练阶段渲染
            else:
                # 始终启用对齐
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=True)
                # for param in motion_net.parameters():
                #     param.requires_grad = True

            image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # 解包渲染结果
            # image_white = render_pkg["render"]  # 渲染图像
            # alpha = render_pkg["alpha"]  # 透明度通道
            # viewspace_point_tensor = render_pkg["viewspace_points"]  # 视图空间点
            # visibility_filter = render_pkg["visibility_filter"]  # 可见性过滤器
            # radii = render_pkg["radii"]  # 高斯点半径

            # 准备GT图像
            gt_image  = viewpoint_cam.original_image.cuda() / 255.0  # 归一化
            gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask   # 背景替换

            # 运动网络冻结（训练后期）
            if iteration > motion_stop_iter:
                for param in motion_net.parameters():
                    param.requires_grad = False # 停止梯度更新

            # 高斯参数冻结（最终阶段）
            if iteration > bg_iter:
                gaussians._xyz.requires_grad = False
                gaussians._opacity.requires_grad = False
                # gaussians._features_dc.requires_grad = False
                # gaussians._features_rest.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False



            # Loss
            if iteration < bg_iter:
                if hair_mask_iter:  # 判断头发掩码更新周期
                    image_white[:, hair_mask] = background[:, None] # 将渲染图像的头发区域像素值替换为背景色
                    gt_image_white[:, hair_mask] = background[:, None]  # 将真实图像的头发区域像素值替换为背景色

                # image_white[:, mouth_mask] = 1
                ## gt_image_white[:, mouth_mask] = background[:, None]

                patch_range = (10,30)  # 10 - 30
                loss_l2_dpt = patch_norm_mse_loss2(image_white[None, ...], gt_image_white[None, ...],
                                                  randint(patch_range[0], patch_range[1]), 0.02)
                patch_img = 0.0004 * loss_l2_dpt
                loss_global = patch_norm_mse_loss_global2(image_white[None, ...], gt_image_white[None, ...],
                                                         randint(patch_range[0], patch_range[1]), 0.02)
                global_img = 0.004 * loss_global

                Ll1 = l1_loss(image_white, gt_image_white)  # 计算L1 Loss
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))   # 组合L1和DSSIM损失
                loss += global_img + patch_img

                # 创建去除头发和嘴巴区域的掩码
                non_hair_mouth_mask = ~(hair_mask | mouth_mask)

                # 应用掩码获取去除头发和嘴巴的区域
                image_white_no_hair_mouth = image_white[:, non_hair_mouth_mask]
                gt_image_white_no_hair_mouth = gt_image_white[:, non_hair_mouth_mask]

                # 计算去除头发和嘴巴区域后的L1损失
                # Ll1_no_hair_mouth = l1_loss(image_white_no_hair_mouth, gt_image_white_no_hair_mouth)
                # 将去除头发和嘴巴区域后的L1损失添加到总损失中，可根据需要调整权重
                # loss += 1 * (Ll1_no_hair_mouth)  # 使用0.1作为示例权重，请根据实际情况调整

                if not mode_long and iteration > warm_step + 2000:  # 短模式+预热后2000次迭代
                    # 法线一致性损失
                    # loss += 0.01 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * render_pkg["normal"]).sum(0)[head_m
                    # 计算图像梯度权重（边缘区域权重低，平坦区域权重大）

                    '''
                    grad_weight = get_img_grad_weight(gt_image)  # 调用梯度权重函数
                    image_weight = torch.exp(-grad_weight).clamp(0, 1).detach() ** 2  # 转换为误差加权系数

                    #法向误差
                    gt_normal = viewpoint_cam.talking_dict["normal"].cuda()
                    pred_normal = render_pkg["normal"]
                    normal_error = (1 - gt_normal * pred_normal).sum(0)

                    # 头部掩码+梯度权重加权
                    weighted_error = (normal_error * image_weight)[head_mask]
                    # 计算最终损失（保持原来的系数0.01）
                    normal_loss = weighted_error.mean()
                    '''
                    render_normal = render_pkg["normal"]
                    gt_normal = viewpoint_cam.talking_dict["normal"].cuda()

                    patch_range = (10, 30)  # 10 - 30
                    loss_l2_dpt = patch_norm_mse_loss2(render_normal[None, ...], gt_normal[None, ...],
                                                       randint(patch_range[0], patch_range[1]), 0.02)
                    patch_normal = 0.002 * loss_l2_dpt
                    loss_global = patch_norm_mse_loss_global2(render_normal[None, ...],  gt_normal[None, ...],
                                                              randint(patch_range[0], patch_range[1]), 0.02)
                    global_normal = 0.02 * loss_global

                    normal_loss = 0.01 * (1 -gt_normal * render_normal ).sum(0)[head_mask].mean()
                    # normal_loss = 0.01 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * render_pkg["normal"]).sum(0)[head_mask].mean()
                    loss += normal_loss # + patch_normal + global_normal

                    depth_normal = depth_to_normal(viewpoint_cam, render_pkg["depth"]).permute(2,0,1)
                    # normalerror = ((render_normal - depth_normal)).abs().sum(0)
                    Ll1norm = l1_loss( viewpoint_cam.talking_dict["normal"].cuda(), depth_normal)  # 计算L1 Loss
                    #loss += 0.001 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * depth_normal).sum(0)[face_mask].mean()
                    #loss += 0.005* Ll1norm
                    if tb_writer is not None:
                        tb_writer.add_scalar(" normal_loss", normal_loss.item(), global_step=iteration)
                    if iteration % opt.opacity_reset_interval > 100:    # 深度一致性损失
                        # depth_normal = depth_to_normal(viewpoint_cam, render_pkg["depth"]).permute(2,0,1)
                        # loss += 0.001 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * depth_normal).sum(0)[face_mask^mouth_mask].mean()

                        depth = render_pkg["depth"][0]  # 渲染深度
                        depth_mono = viewpoint_cam.talking_dict['depth'].cuda() # 单目估计深度

                        # 归一化深度差异损失
                        # loss += 1e-2 * (normalize(depth)[face_mask^mouth_mask] - normalize(depth_mono)[face_mask^mouth_mask]).abs().mean()

                        depth_loss= 1e-2 * (normalize(depth)[face_mask] - normalize(depth_mono)[face_mask]).abs().mean()
                        loss += depth_loss
                        if tb_writer is not None:
                            tb_writer.add_scalar("depth_loss", depth_loss.item(), global_step=iteration)

                # mouth_alpha_loss = 1e-2 * (alpha[:,mouth_mask]).mean()
                # if not torch.isnan(mouth_alpha_loss):
                    # loss += mouth_alpha_loss
                # print(alpha[:,mouth_mask], mouth_mask.sum())

                #运动变化正则化
                if iteration > warm_step:
                    loss += 1e-4 * (render_pkg['motion']['d_xyz'].abs()).mean()  # 位移变化正则
                    loss += 1e-4 * (render_pkg['motion']['d_rot'].abs()).mean()  # 旋转变化正则
                    loss += 1e-4 * (render_pkg['motion']['d_opa'].abs()).mean()  # 透明度变化正则
                    loss += 1e-4 * (render_pkg['motion']['d_scale'].abs()).mean()  # 尺度变化正则
                    loss += 1e-5 * (render_pkg['p_motion']['p_xyz'].abs()).mean()  # 个性化位移正则

                    # 透明度正则化。 确保头部区域内不透明，区域外透明
                    opacity_loss = 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())
                    loss += opacity_loss

                    if tb_writer is not None:
                        tb_writer.add_scalar("opacity_loss", opacity_loss.item(), global_step=iteration)

                    [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']  # 嘴唇矩形区域
                    attn_lips= 1e-4 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean() # 增强嘴部注意力
                    loss += attn_lips
                    if tb_writer is not None:
                        tb_writer.add_scalar("attn_lips", attn_lips.item(), global_step=iteration)

                    if not hair_mask_iter:  # 非头发掩码更新时
                        attn_aud= 1e-4 * (render_pkg["attn"][1][hair_mask]).mean()    # 音频注意力
                        loss += attn_aud
                        attn_exp= 1e-4 * (render_pkg["attn"][0][hair_mask]).mean()    # 表情注意力
                        loss += attn_exp
                        if tb_writer is not None:
                            tb_writer.add_scalar("attn_aud", attn_aud.item(), global_step=iteration)
                            tb_writer.add_scalar("attn_exp", attn_exp.item(), global_step=iteration)
                    # loss += l2_loss(image_white[:, xmin:xmax, ymin:ymax], image_white[:, xmin:xmax, ymin:ymax])

                # Depth Loss
                #  depth
                if iteration > 1000:
                    # depth_loss_soft = torch.tensor(0.).cuda()
                    #render_pkg_opa = render_motion_opa(viewpoint_cam, gaussians, motion_net, pipe, background,
                    #                                   return_attn=True,
                    #                                   personalized=False, align=True)

                    # depth, alpha = render_pkg_opa["depth"], render_pkg_opa["alpha"]
                    depth, alpha = render_pkg["depth"], render_pkg["alpha"]
                    depth_mono = viewpoint_cam.talking_dict['depth'].cuda() # 单目估计深度
                    depth_mono[~(face_mask+hair_mask)] = 0
                    if iteration % 2000 == 0:  # 保存深度图
                        save_depth_as_png(depth, "depth_map", iteration, map_type="opa")
                        save_depth_as_png(depth_mono, "depth_map", iteration, map_type="mono")
                    patch_range = (10, 30)  #10 - 30
                    loss_l2_dpt = patch_norm_mse_loss(depth[None, ...], depth_mono[None, ...],
                                                      randint(patch_range[0], patch_range[1]), 0.02)
                    patch_depth= 0.0002 * loss_l2_dpt

                    # if iteration > 10000:
                    #    depth_loss_soft += 0.00004 * loss_depth_smoothness(depth[None, ...], gt_depth[None, ...])

                    loss_global = patch_norm_mse_loss_global(depth[None, ...], depth_mono[None, ...],
                                                             randint(patch_range[0], patch_range[1]), 0.02)
                    global_depth= 0.002 * loss_global

                    loss += patch_depth + global_depth
                    if tb_writer is not None:
                        tb_writer.add_scalar("patch_depth", patch_depth.item(), global_step=iteration)
                        tb_writer.add_scalar("global_depth", global_depth.item(), global_step=iteration)

                image_t = image_white.clone()
                gt_image_t = gt_image_white.clone()

                total_loss = loss

                # ================= [Stream 2: TimeFormer Branch] =================
                if iteration > warm_step:
                    loss_tf_accum = 0.0

                    # 1. 准备归一化的空间坐标
                    # K-Planes 的 Grid 范围是 [-1, 1]，必须将世界坐标归一化
                    # scene.cameras_extent 是场景半径，用于将点云缩放到单位球内
                    scene_radius = scene.cameras_extent
                    if scene_radius < 0.1: scene_radius = 1.0  # 防止除零或异常小

                    # 归一化并截断，防止越界导致 Grid 采样到无效值
                    # 注意：通常不需要对输入的 XYZ 求导 (detach)，我们只训练 TimeFormer 的 Grid
                    norm_xyz = (gaussians.get_xyz.detach() / scene_radius).clamp(-1.0, 1.0)

                    # 2. 准备 Batch 输入: [Norm_XYZ, Norm_T]
                    tf_inputs = []
                    for cam in viewpoint_cams:
                        # 归一化时间 t -> [-1, 1]
                        norm_t = cam.uid / (total_frames - 1)
                        norm_t = 2 * norm_t - 1

                        # 扩展时间标量: [N, 1]
                        t_embed = torch.ones((gaussians.get_xyz.shape[0], 1), device="cuda") * norm_t

                        # [关键] 拼接归一化后的 XYZ 和 T
                        tf_input = torch.cat([norm_xyz, t_embed], dim=1)
                        tf_inputs.append(tf_input.unsqueeze(0))

                    # 拼接成 Batch Tensor: [B, N, 4]
                    time_ori = torch.cat(tf_inputs, dim=0)

                    # 3. TimeFormer 前向传播 -> 得到 Offsets
                    full_time_offsets = timeFormer(time_ori)  # [B, N, 3]
                    # 3. 遍历 Batch 计算 Masked L1 Loss
                    for i, cam in enumerate(viewpoint_cams):
                        offset_i = full_time_offsets[i]

                        # 自定义渲染 (XYZ + Offset -> MotionNet -> Image)
                        render_pkg_tf = render_timeformer(cam, gaussians, motion_net, offset_i, pipe, background)
                        image_tf = render_pkg_tf["render"]

                        # 背景合成 (与 Stream 1 保持一致)
                        alpha_tf = render_pkg_tf["alpha"]
                        image_tf = image_tf - background[:, None, None] * (
                                    1.0 - alpha_tf) + cam.background.cuda() / 255.0 * (1.0 - alpha_tf)

                        # 获取 GT
                        gt_image = cam.original_image.cuda() / 255.0

                        # 获取 Mask (用于只计算人脸区域 Loss)
                        if i == 0:
                            # 主相机可以直接复用上面算好的 mask 变量
                            curr_mask = head_mask
                        else:
                            # 辅助相机需要从 talking_dict 重新获取
                            f_mask = torch.as_tensor(cam.talking_dict["face_mask"]).cuda()
                            h_mask = torch.as_tensor(cam.talking_dict["hair_mask"]).cuda()
                            m_mask = torch.as_tensor(cam.talking_dict["mouth_mask"]).cuda()
                            curr_mask = f_mask + h_mask + m_mask

                        # 计算 Masked L1 Loss
                        # 只优化头部区域，忽略背景
                        loss_tf_accum += l1_loss(image_tf * curr_mask, gt_image * curr_mask)

                    # 4. 平均 Loss 并加权
                    loss_tf_mean = loss_tf_accum / len(viewpoint_cams)
                    total_loss += weight_t_loss * loss_tf_mean

                    # 5. Offset 正则化 (防止偏移过大)
                    total_loss += weight_tf_reg * l1_loss(full_time_offsets, torch.zeros_like(full_time_offsets))

                # =================================================================



            else:   # 超过bg_iter后切换到真实背景
                # with real bg
                # 合成真实背景图像
                image = image_white - background[:, None, None] * (1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

                # 计算真实背景下的损失
                Ll1 = l1_loss(image, gt_image)
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                image_t = image.clone()
                gt_image_t = gt_image.clone()

            if iteration > lpips_start_iter:   # LPIPS损失启动点
                # mask mouth
                # 嘴部区域掩码
                [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                if mode_long:   # 长训练模式嘴部强化
                    loss += 0.01 * lpips_criterion(image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1, gt_image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1).mean()

                # 嘴部区域置背景
                # image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
                # gt_image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]

                # 随机补丁LPIPS损失
                patch_size = random.randint(32, 48) * 2 # 64-96像素补丁
                if mode_long:   # 长训练模式增强
                    loss += 0.2 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()
                # 基础LPIPS损失
                lpips_loss= 0.01 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()
                loss += lpips_loss
                if tb_writer is not None:
                    tb_writer.add_scalar("lpips_loss", lpips_loss.item(), global_step=iteration)

                # loss += 0.5 * lpips_criterion(image_t[None, ...] * 2 - 1, gt_image_t[None, ...] * 2 - 1).mean()


            loss_records['iterations'].append(iteration)

        else:
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # 渲染面部模型：render_motion
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, align=True)
            # 提取渲染结果的关键信息
            viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"]

            # 透明度提取
            alpha = render_pkg["alpha"]
            image = render_pkg["render"] - background[:, None, None] * (
                        1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

            # 获取归一化真实图像
            gt_image = viewpoint_cam.original_image.cuda() / 255.0
            # 创建带背景的真实图像（头部区域保留，其他区域置为背景色）
            gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

            # 参数冻结控制
            if iteration > bg_iter:
                for param in motion_net.parameters():
                    param.requires_grad = False

                # 冻结几何参数梯度
                gaussians._xyz.requires_grad = False
                gaussians._scaling.requires_grad = False
                gaussians._rotation.requires_grad = False


            # Loss
            # 损失计算（绿幕阶段）
            if iteration < bg_iter:
                image[:, ~head_mask] = background[:, None]  # 非头部区域置为背景色

                Ll1 = l1_loss(image, gt_image_white)  # 计算L1损失
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image_white))  # 添加DSSIM结构损失
                loss += 1e-3 * (((1 - alpha) * head_mask).mean() + (alpha * ~head_mask).mean())  # 添加透明度正则项

                # 克隆图像
                image_t = image.clone()
                gt_image_t = gt_image_white.clone()

            # 损失计算（真实背景阶段）
            else:
                Ll1 = l1_loss(image, gt_image)  # 直接计算L1损失
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  # 添加DSSIM结构损失

                # 克隆图像
                image_t = image.clone()
                gt_image_t = gt_image.clone()

            if iteration > lpips_start_iter:  # 迭代超过总迭代数一半
                patch_size = random.randint(16, 21) * 2  # 随机生成补丁大小（32-42像素）
                # 计算补丁级LPIPS感知损失    加权(0.05)加入总损失
                loss += 0.05 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size),
                                               patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()



        loss.backward()  # 反向传播计算梯度
        iter_end.record()  # 记录迭代结束时间

        with torch.no_grad():
            # Progress bar
            # 进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log   # 更新指数移动平均损失
            if iteration % 2 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"}) # , "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"
                progress_bar.update(2)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 训练报告生成
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion, (pipe, background))
            if (iteration in saving_iterations):  # 在保存点迭代
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration) + '_face')  # 保存高斯模型

            if (iteration in checkpoint_iterations):  # 在检查点迭代
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # 准备检查点数据
                ckpt = (
                    gaussians.capture(),  # 捕获高斯模型状态
                    motion_net.state_dict(),  # 运动网络参数
                    motion_optimizer.state_dict(),  # 优化器状态
                    iteration  # 当前迭代次数
                )
                # 保存带迭代编号的检查点
                torch.save(ckpt, scene.model_path + "/chkpnt_" + str(iteration) + ".pth")
                # 保存最新检查点
                torch.save(ckpt, scene.model_path + "/chkpnt_latest" + ".pth")
                torch.save(loss_records,  scene.model_path + "/loss_records" + ".pth")

            # Densification 密度控制
            if iteration < opt.densify_until_iter:  # 在密度控制阶段
                # Keep track of max radii in image-space for pruning
                # 更新最大半径（用于剪枝）
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"])    # 添加pixels

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:   # 密度化与剪枝
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

                if not mode_long:   # 短训练模式
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # bg prune 背景剪枝
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh

                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                gaussians.prune_points(bg_color_mask.squeeze())

                if not mode_long:   # 短训练模式额外剪枝
                    gaussians.prune_points((gaussians.get_xyz[:, -1] < -0.07).squeeze())    # 剪除深度过小的点


            # Optimizer step    优化器步骤
            if iteration <= opt.iterations:
                if iteration <= FACE_STAGE_ITER:
                    motion_optimizer.step()
                    gaussians.optimizer.step()
                    motion_optimizer.zero_grad()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    scheduler.step()
                else:
                    # 执行优化步骤
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)



def prepare_output_and_logger(args):    # 输出目录与日志
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        # args.model_path = os.path.join("/root/autodl-tmp/logs", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, renderFunc, renderArgs):
    if tb_writer:  # TensorBoard可用时
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)  # L1损失
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)  # 总损失
        tb_writer.add_scalar('iter_time', elapsed, iteration)  # 迭代耗时

    # Report test and samples of training set
    # 报告测试情况以及训练集样本
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 创建验证集配置   测试集5-100帧每5帧取1帧     训练集5-30帧每5帧取1帧
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 5)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 创建存储图像的根目录
        report_root = os.path.join('training_reports/train', f'iter_{iteration}')
        os.makedirs(report_root, exist_ok=True)
        for config in validation_configs:   # 遍历测试/训练配置
            if config['cameras'] and len(config['cameras']) > 0:    # 检查相机数据存在
                l1_test = 0.0
                psnr_test = 0.0

                # 遍历配置中的每个相机视角
                for idx, viewpoint in enumerate(config['cameras']):
                    if renderFunc is render:     # 基础渲染
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:   # 运动渲染
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, align=True, *renderArgs)

                    # 图像后处理
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)  # 裁剪到[0,1]
                    alpha = render_pkg["alpha"]  # 透明度通道
                    normal = render_pkg["normal"] * 0.5 + 0.5   #法线图处理 归一化到[0,1]

                    # 深度图处理 归一化
                    depth = render_pkg["depth"] * alpha + (render_pkg["depth"] * alpha).mean() * (1 - alpha)
                    depth = (depth - depth.min()) / (depth.max() - depth.min())

                    # 深度法线图生成
                    depth_normal = depth_to_normal(viewpoint, render_pkg["depth"]).permute(2, 0, 1)  # 深度转法线
                    depth_normal = depth_normal * alpha.detach()  # 应用透明度
                    depth_normal = depth_normal * 0.5 + 0.5  # 归一化到[0,1]

                    # 背景合成
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0) # GT图像

                    # === 添加图像保存代码 ===
                    # 创建配置特定目录
                    config_name = config['name']
                    config_dir = os.path.join(report_root, config_name)
                    os.makedirs(config_dir, exist_ok=True)

                    # 安全文件名
                    safe_name = viewpoint.image_name.replace('/', '_')

                    # 保存渲染图像
                    render_img = image.permute(1, 2, 0).cpu().numpy()
                    render_img = (render_img * 255).astype(np.uint8)
                    render_path = os.path.join(config_dir, f"{safe_name}_render.png")
                    Image.fromarray(render_img).save(render_path)

                    # 保存GT图像
                    gt_img = gt_image.permute(1, 2, 0).cpu().numpy()
                    gt_img = (gt_img * 255).astype(np.uint8)
                    gt_path = os.path.join(config_dir, f"{safe_name}_gt.png")
                    Image.fromarray(gt_img).save(gt_path)
                    # === 结束添加代码 ===


                    # 嘴部掩码处理
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()  # 形态学闭运算

                    if tb_writer and (idx < 5): # 只记录前5个视角
                        # 基础渲染结果
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        # 真实图像
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        # 深度图
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        # 嘴部掩码可视化
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name), (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name), (~mouth_mask[None] * gt_image)[None], global_step=iteration)
                        # 法线图
                        tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), normal[None], global_step=iteration)
                        # 深度法线图
                        tb_writer.add_images(config['name'] + "_view_{}/normal_from_depth".format(viewpoint.image_name), depth_normal[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/normal_mono".format(viewpoint.image_name), (viewpoint.talking_dict["normal"]*0.5+0.5)[None], global_step=iteration)
                        # if config['name']=="train":
                        #     depth_mono = 1.0 - viewpoint.talking_dict['depth'].cuda()
                        #     tb_writer.add_images(config['name'] + "_view_{}/depth_mono".format(viewpoint.image_name), depth_mono[None, None], global_step=iteration)

                        # 运动渲染特有可视化
                        if renderFunc is not render:
                            # 音频注意力图
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name), (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None], global_step=iteration)
                            # 表情注意力图
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name), (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None], global_step=iteration)

                    # 计算当前视角指标
                    l1_test += l1_loss(image, gt_image).mean().double()  # L1损失
                    psnr_test += psnr(image, gt_image).mean().double()  # PSNR

                # 计算平均指标
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)  # 不透明度直方图
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)   # 高斯点数量统计
        torch.cuda.empty_cache()

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img


def save_depth_as_png(depth_map, folder_name, iteration, map_type):
    """
    将深度图保存为 PNG 文件，并根据 map_type 分类存放在对应的文件夹中。
    :param depth_map: 需要保存的深度图 (torch.Tensor)
    :param folder_name: 基本文件夹路径，例如 'depth_map'
    :param iteration: 当前的迭代次数，用于文件命名
    :param map_type: 'opa' 或 'depth'，用于指定保存的深度图类型
    """
    # 将深度图从计算图中分离，并转换为 numpy 数组
    depth_map_np = depth_map.detach().squeeze().cpu().numpy()

    # 将深度值归一化到 0-1 的范围
    depth_map_normalized = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())

    # 创建对应类型的文件夹路径
    save_path = os.path.join(folder_name, map_type)
    os.makedirs(save_path, exist_ok=True)

    # 生成文件名并保存
    file_name = f"depth_map_iter_{iteration}.png"
    file_path = os.path.join(save_path, file_name)
    plt.imsave(file_path, depth_map_normalized, cmap='gray')
    print(f"Depth map saved as {file_path}")


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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--long", action='store_true', default=False)
    parser.add_argument("--pretrain_path", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.long, args.pretrain_path)

    # All done
    print("\nTraining complete.")
