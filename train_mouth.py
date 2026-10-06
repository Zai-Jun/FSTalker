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

import os
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim
from gaussian_renderer import render, render_motion, render_motion_mouth
import sys
from scene import Scene, GaussianModel, MouthMotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import loadCamOnTheFly
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    testing_iterations = [i for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations]

    # vars
    warm_step = 3000
    bg_iter = opt.iterations-1000 # opt.densify_until_iter
    lpips_start_iter = bg_iter
    motion_stop_iter = bg_iter
    mouth_select_iter = bg_iter - 10000
    mouth_step = 1 / mouth_select_iter
    select_interval = 7

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    motion_net = MouthMotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / bg_iter))

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)

    bg_color = [0, 1, 0] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

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

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # find a big mouth

        au_global_lb = viewpoint_cam.talking_dict['au25'][1]
        au_global_ub = viewpoint_cam.talking_dict['au25'][4]
        au_window = (au_global_ub - au_global_lb) * 0.2

        au_ub = au_global_ub
        au_lb = au_ub - mouth_step * iteration * (au_global_ub - au_global_lb)

        if iteration < warm_step:
            while viewpoint_cam.talking_dict['au25'][0] < au_global_ub:
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if warm_step < iteration < mouth_select_iter:
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['au25'][0] < au_lb or viewpoint_cam.talking_dict['au25'][0] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            if viewpoint_cam.original_image == None:
                viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

            while torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda().sum() < 20:
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                if viewpoint_cam.original_image == None:
                    viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

        if viewpoint_cam.original_image == None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration > bg_iter:
            # turn to black
            bg_color = [0, 0, 0] # if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask =  face_mask + hair_mask
        
        [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
        lips_mask = torch.zeros_like(mouth_mask)
        lips_mask[xmin:xmax, ymin:ymax] = True

        if iteration < warm_step:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        else:
            render_pkg = render_motion_mouth(viewpoint_cam, gaussians, motion_net, pipe, background)

        image_green, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image  = viewpoint_cam.original_image.cuda() / 255.0
        gt_image_green = gt_image * mouth_mask + background[:, None, None] * ~mouth_mask

        if iteration > motion_stop_iter:
            for param in motion_net.parameters():
                param.requires_grad = False
        if iteration > bg_iter:
            gaussians._xyz.requires_grad = False
            gaussians._opacity.requires_grad = False
            # gaussians._features_dc.requires_grad = False
            # gaussians._features_rest.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False
        
        # Loss            
        image_green[:, (lips_mask ^ mouth_mask)] = background[:, None]

        Ll1 = l1_loss(image_green, gt_image_green)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_green, gt_image_green))


        if iteration > warm_step:
            # loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
            loss += 1e-3 * (((1-alpha) * lips_mask).mean() + (alpha * ~lips_mask).mean())

        image_t = image_green.clone()
        gt_image_t = gt_image_green.clone()

        if iteration > lpips_start_iter:
            patch_size = random.randint(16, 21) * 2
            loss += 0.5 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()



        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration)+'_mouth')

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion_mouth, (pipe, background))
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, scene.model_path + "/chkpnt_mouth_" + str(iteration) + ".pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_mouth_latest" + ".pth")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

                    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                    dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    from utils.sh_utils import eval_sh
                    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                    bg_color_mask = (colors_precomp[..., 0] < 20/255) * (colors_precomp[..., 1] > 235/255) * (colors_precomp[..., 2] < 20/255)
                    gaussians.xyz_gradient_accum[bg_color_mask] /= 2
                    gaussians._opacity[bg_color_mask] = gaussians.inverse_opacity_activation(torch.ones_like(gaussians._opacity[bg_color_mask]) * 0.1)
                    gaussians._scaling[bg_color_mask] /= 10

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                motion_optimizer.step()
                gaussians.optimizer.step()

                motion_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none = True)

                scheduler.step()



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    '''
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    '''
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 10)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if viewpoint.original_image == None:
                        viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))
                        
                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)


                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")







# ================   FLAME初始 (全局仅执行一次)backup   =====
if iteration == 1:
    with torch.no_grad():
        codedict = {}
        codedict['shape'] = viewpoint_cam.talking_dict['shape']
        DeformModel.example_init(codedict)

        codedict['expr'] = viewpoint_cam.talking_dict['exp_param']
        codedict['eyes_pose'] = viewpoint_cam.talking_dict['eyes_pose']
        codedict['eyelids'] = viewpoint_cam.talking_dict['eyelids']
        codedict['jaw_pose'] = viewpoint_cam.talking_dict['jaw_pose']

        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

    verts_flame = verts_final[0].detach()

    # ========================================================
    # 空间坐标系绝对对齐 (包含 尺度归一化 与 内参畸变校准)
    # ========================================================
    import math

    ckpt_path = os.path.join(args.source_path, "checkpoint", f"{int(viewpoint_cam.uid):05d}.frame")
    payload = torch.load(ckpt_path)

    # 1. MICA World 到 MICA Camera (直接使用，无需逆转)
    R_m = torch.tensor(payload['opencv']['R'][0], device="cuda", dtype=torch.float32)
    T_m = torch.tensor(payload['opencv']['t'][0], device="cuda", dtype=torch.float32)
    # 矩阵乘法：X_cam = X_w @ R^T + T
    verts_cam_m = torch.matmul(verts_flame, R_m.T) + T_m

    # 2. 内参校准 (Intrinsic Alignment)
    K_mica = torch.tensor(payload['opencv']['K'][0], device="cuda", dtype=torch.float32)
    fx_m, fy_m = K_mica[0, 0], K_mica[1, 1]
    cx_m, cy_m = K_mica[0, 2], K_mica[1, 2]

    # 获取 TG 真实图像分辨率
    if hasattr(viewpoint_cam, 'width') and hasattr(viewpoint_cam, 'height'):
        W, H = int(viewpoint_cam.width), int(viewpoint_cam.height)
    else:
        from PIL import Image

        with Image.open(viewpoint_cam.image_path) as _img:
            W, H = _img.size

    fx_tg = W / (2 * math.tan(viewpoint_cam.FoVx / 2.0))
    fy_tg = H / (2 * math.tan(viewpoint_cam.FoVy / 2.0))
    cx_tg, cy_tg = W / 2.0, H / 2.0

    # 在相机空间中对齐透视关系
    X_adj = verts_cam_m[:, 0] * (fx_m / fx_tg) + verts_cam_m[:, 2] * ((cx_m - cx_tg) / fx_tg)
    Y_adj = verts_cam_m[:, 1] * (fy_m / fy_tg) + verts_cam_m[:, 2] * ((cy_m - cy_tg) / fy_tg)
    Z_adj = verts_cam_m[:, 2]
    verts_cam_adj = torch.stack([X_adj, Y_adj, Z_adj], dim=1)

    # 3. 场景尺度归一化 (Scale Alignment)
    # 计算 MICA(米) 到 TG(归一化单位) 的绝对空间缩放比例
    C_m = -torch.matmul(R_m.T, T_m)  # MICA 真实相机中心
    dist_m = torch.norm(C_m)
    dist_tg = torch.norm(viewpoint_cam.camera_center)
    scale_factor = dist_tg / dist_m

    verts_cam_tg = verts_cam_adj * scale_factor

    # ========================================================
    # [终极双重自适应对齐] 基于解剖学锚点的 偏转角(Roll) 与 边界位移 补偿
    # ========================================================
    tg_lmks = viewpoint_cam.talking_dict.get('lmks', None)

    if tg_lmks is not None:
        tg_lmks = tg_lmks.to("cuda")

        # ----------------------------------------------------
        # 第一道锁：偏转角补偿 (Roll Lock)
        # ----------------------------------------------------
        # 1. 提取 GT 的绝对解剖学垂直中线：鼻梁顶部(27) -> 下巴底端(8)
        vec_gt_x = tg_lmks[8, 0] - tg_lmks[27, 0]
        vec_gt_y = tg_lmks[8, 1] - tg_lmks[27, 1]
        # 确保向量在屏幕空间向下
        if vec_gt_y < 0:
            vec_gt_x, vec_gt_y = -vec_gt_x, -vec_gt_y
        angle_gt = torch.atan2(vec_gt_y, vec_gt_x)

        # 2. 提取 3D 点云的垂直中线：利用 Canonical 空间下 Y 轴的极值点
        # 使用 .item() 转换为原生整型，避免多维索引解析歧义
        idx_top = verts_flame[:, 1].argmax().item()
        idx_bottom = verts_flame[:, 1].argmin().item()

        # 显式对维度 0 进行列表索引
        pts_cam = verts_cam_tg[[idx_top, idx_bottom], :]
        u_pts = pts_cam[:, 0] / pts_cam[:, 2] * fx_tg + cx_tg
        v_pts = pts_cam[:, 1] / pts_cam[:, 2] * fy_tg + cy_tg

        vec_proj_x = u_pts[1] - u_pts[0]
        vec_proj_y = v_pts[1] - v_pts[0]
        if vec_proj_y < 0:
            vec_proj_x, vec_proj_y = -vec_proj_x, -vec_proj_y
        angle_proj = torch.atan2(vec_proj_y, vec_proj_x)

        # 3. 计算偏转角差值并在相机空间执行 2D 旋转
        delta_angle = angle_gt - angle_proj

        c_x = verts_cam_tg[:, 0].mean()
        c_y = verts_cam_tg[:, 1].mean()
        x_shifted = verts_cam_tg[:, 0] - c_x
        y_shifted = verts_cam_tg[:, 1] - c_y

        cos_a = torch.cos(delta_angle)
        sin_a = torch.sin(delta_angle)

        verts_cam_tg[:, 0] = x_shifted * cos_a - y_shifted * sin_a + c_x
        verts_cam_tg[:, 1] = x_shifted * sin_a + y_shifted * cos_a + c_y

        # ----------------------------------------------------
        # 第二道锁：下巴边界锚定 (Chin-Anchor Shift)
        # ----------------------------------------------------
        # 旋转后重新全量投影
        u_proj = verts_cam_tg[:, 0] / verts_cam_tg[:, 2] * fx_tg + cx_tg
        v_proj = verts_cam_tg[:, 1] / verts_cam_tg[:, 2] * fy_tg + cy_tg

        # 水平对齐：依然使用 X 轴整体质心（因为脸部左右对称，X轴不受脑壳高度影响）
        delta_u = tg_lmks[:, 0].mean() - u_proj.mean()

        # 垂直对齐：【核心纠正】废除 Y 轴质心，严格使用下巴绝对最低点！
        gt_chin_v = tg_lmks[8, 1]  # GT 第 8 个点是确切的下巴底线
        proj_chin_v = v_proj.max()  # 去掉脖子后的 3D 模型在屏幕上的最大 Y 值必定是下巴
        delta_v = gt_chin_v - proj_chin_v

        # 逆透视解算与施加补偿
        mean_z = verts_cam_tg[:, 2].mean()
        verts_cam_tg[:, 0] += (delta_u * mean_z / fx_tg)
        verts_cam_tg[:, 1] += (delta_v * mean_z / fy_tg)

        print(f"[对齐成功] 双重锁校验通过 | 修复偏转: {delta_angle.item() * 180 / math.pi:.2f}度")
    else:
        print("\n[系统致命警告] talking_dict 中缺失 lmks，自适应对齐失效！")
    # ========================================================

    # 4. TG Camera 反推回 TG World
    R_tg = torch.tensor(viewpoint_cam.R, device="cuda", dtype=torch.float32)
    T_tg = torch.tensor(viewpoint_cam.T, device="cuda", dtype=torch.float32)

    # X_cam = X_w @ R_tg + T_tg  =>  X_w = (X_cam - T_tg) @ R_tg_inv
    init_verts_aligned = torch.matmul(verts_cam_tg - T_tg, torch.inverse(R_tg))

    # ========================================================

    # 初始化高斯场并固化叶子节点
    gaussians.create_from_verts(init_verts_aligned)
    gaussians.training_setup(opt)




    # ================   FLAME初始 (全局仅执行一次) BACK UP2  =====
    if iteration == 1:
        with torch.no_grad():
            codedict = {}
            codedict['shape'] = viewpoint_cam.talking_dict['shape']
            DeformModel.example_init(codedict)

            codedict['expr'] = viewpoint_cam.talking_dict['exp_param']
            codedict['eyes_pose'] = viewpoint_cam.talking_dict['eyes_pose']
            codedict['eyelids'] = viewpoint_cam.talking_dict['eyelids']
            codedict['jaw_pose'] = viewpoint_cam.talking_dict['jaw_pose']

            verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

        verts_flame = verts_final[0].detach()

        # ========================================================
        # 空间坐标系绝对对齐 (包含 尺度归一化 与 内参畸变校准)
        # ========================================================
        import math
        ckpt_path = os.path.join(args.source_path, "checkpoint", f"{int(viewpoint_cam.uid):05d}.frame")
        payload = torch.load(ckpt_path)

        # 1. MICA World 到 MICA Camera (直接使用，无需逆转)
        R_m = torch.tensor(payload['opencv']['R'][0], device="cuda", dtype=torch.float32)
        T_m = torch.tensor(payload['opencv']['t'][0], device="cuda", dtype=torch.float32)
        # 矩阵乘法：X_cam = X_w @ R^T + T
        verts_cam_m = torch.matmul(verts_flame, R_m.T) + T_m

        # 2. 内参校准 (Intrinsic Alignment)
        K_mica = torch.tensor(payload['opencv']['K'][0], device="cuda", dtype=torch.float32)
        fx_m, fy_m = K_mica[0, 0], K_mica[1, 1]
        cx_m, cy_m = K_mica[0, 2], K_mica[1, 2]

        # 获取 TG 真实图像分辨率
        if hasattr(viewpoint_cam, 'width') and hasattr(viewpoint_cam, 'height'):
            W, H = int(viewpoint_cam.width), int(viewpoint_cam.height)
        else:
            from PIL import Image
            with Image.open(viewpoint_cam.image_path) as _img:
                W, H = _img.size

        fx_tg = W / (2 * math.tan(viewpoint_cam.FoVx / 2.0))
        fy_tg = H / (2 * math.tan(viewpoint_cam.FoVy / 2.0))
        cx_tg, cy_tg = W / 2.0, H / 2.0

        # 在相机空间中对齐透视关系
        X_adj = verts_cam_m[:, 0] * (fx_m / fx_tg) + verts_cam_m[:, 2] * ((cx_m - cx_tg) / fx_tg)
        Y_adj = verts_cam_m[:, 1] * (fy_m / fy_tg) + verts_cam_m[:, 2] * ((cy_m - cy_tg) / fy_tg)
        Z_adj = verts_cam_m[:, 2]
        verts_cam_adj = torch.stack([X_adj, Y_adj, Z_adj], dim=1)

        # 3. 场景尺度归一化 (Scale Alignment)
        # 计算 MICA(米) 到 TG(归一化单位) 的绝对空间缩放比例
        C_m = -torch.matmul(R_m.T, T_m)  # MICA 真实相机中心
        dist_m = torch.norm(C_m)
        dist_tg = torch.norm(viewpoint_cam.camera_center)
        scale_factor = dist_tg / dist_m

        verts_cam_tg = verts_cam_adj * scale_factor

        # ========================================================
        # 姿态与位置对齐 (引入三维刚性变换)
        # ========================================================
        tg_lmks = viewpoint_cam.talking_dict.get('lmks', None)

        if tg_lmks is not None:
            tg_lmks = tg_lmks.to("cuda")

            # 1. 修正手性镜像 (解决 Yaw 偏航角反向问题)
            verts_cam_tg[:, 0] *= -1

            # 2. 三维旋转校准 (修复 Roll 角偏差)
            # 提取 GT 垂直中线
            vec_gt_x = tg_lmks[8, 0] - tg_lmks[27, 0]
            vec_gt_y = tg_lmks[8, 1] - tg_lmks[27, 1]
            if vec_gt_y < 0:
                vec_gt_x, vec_gt_y = -vec_gt_x, -vec_gt_y
            angle_gt = torch.atan2(vec_gt_y, vec_gt_x)

            # 提取 3D 点云垂直轴投影
            idx_top = verts_flame[:, 1].argmax().item()
            idx_bottom = verts_flame[:, 1].argmin().item()

            pts_cam = verts_cam_tg[[idx_top, idx_bottom], :]
            u_pts = pts_cam[:, 0] / pts_cam[:, 2] * fx_tg + cx_tg
            v_pts = pts_cam[:, 1] / pts_cam[:, 2] * fy_tg + cy_tg

            vec_proj_x = u_pts[1] - u_pts[0]
            vec_proj_y = v_pts[1] - v_pts[0]
            if vec_proj_y < 0:
                vec_proj_x, vec_proj_y = -vec_proj_x, -vec_proj_y
            angle_proj = torch.atan2(vec_proj_y, vec_proj_x)

            # 构建绕 Z 轴的三维旋转矩阵并在相机空间执行
            delta_angle = angle_gt - angle_proj
            cos_a = torch.cos(delta_angle)
            sin_a = torch.sin(delta_angle)

            c_x = verts_cam_tg[:, 0].mean()
            c_y = verts_cam_tg[:, 1].mean()

            x_shifted = verts_cam_tg[:, 0] - c_x
            y_shifted = verts_cam_tg[:, 1] - c_y

            verts_cam_tg[:, 0] = x_shifted * cos_a - y_shifted * sin_a + c_x
            verts_cam_tg[:, 1] = x_shifted * sin_a + y_shifted * cos_a + c_y

            # 3. 三维平移校准 (逆透视下巴边界锚定)
            # 重新计算旋转后的投影
            u_proj = verts_cam_tg[:, 0] / verts_cam_tg[:, 2] * fx_tg + cx_tg
            v_proj = verts_cam_tg[:, 1] / verts_cam_tg[:, 2] * fy_tg + cy_tg

            delta_u = tg_lmks[:, 0].mean() - u_proj.mean()
            delta_v = tg_lmks[8, 1] - v_proj.max()

            mean_z = verts_cam_tg[:, 2].mean()
            verts_cam_tg[:, 0] += delta_u * mean_z / fx_tg
            verts_cam_tg[:, 1] += delta_v * mean_z / fy_tg

            print(f"[参数补偿] 偏转角修复计算完成，旋转偏移量: {delta_angle.item() * 180 / math.pi:.2f} 度")
        else:
            print("\n[运行异常] talking_dict 中缺失 lmks，位姿校准已中止。")
        # ========================================================

        # 4. TG Camera 反推回 TG World 并固化叶子节点
        R_tg = torch.tensor(viewpoint_cam.R, device="cuda", dtype=torch.float32)
        T_tg = torch.tensor(viewpoint_cam.T, device="cuda", dtype=torch.float32)

        init_verts_aligned = torch.matmul(verts_cam_tg - T_tg, torch.inverse(R_tg))

        gaussians.create_from_verts(init_verts_aligned)
        gaussians.training_setup(opt)





        # ================   FLAME初始 (全局仅执行一次)  backup3   =====
        if iteration == 1:
            with torch.no_grad():  # 阻断梯度，确保初始化的绝对纯净，防止反向传播污染基底
                codedict = {}
                codedict['shape'] = viewpoint_cam.talking_dict['shape']
                DeformModel.example_init(codedict)

                codedict['expr'] = viewpoint_cam.talking_dict['exp_param']
                codedict['eyes_pose'] = viewpoint_cam.talking_dict['eyes_pose']
                codedict['eyelids'] = viewpoint_cam.talking_dict['eyelids']
                codedict['jaw_pose'] = viewpoint_cam.talking_dict['jaw_pose']

                verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

            verts_flame = verts_final[0].detach()  # 剥离计算图，获取纯数值张量

            # ========================================================
            # 空间坐标系绝对对齐 (包含 尺度归一化 与 内参畸变校准)
            # ========================================================

            ckpt_path = os.path.join(args.source_path, "checkpoint", f"{int(viewpoint_cam.uid):05d}.frame")
            payload = torch.load(ckpt_path)

            # 1. MICA World 到 MICA Camera
            # R_m: MICA 的旋转矩阵 (3x3); T_m: MICA 的平移向量 (3,)
            R_m = torch.tensor(payload['opencv']['R'][0], device="cuda", dtype=torch.float32)
            T_m = torch.tensor(payload['opencv']['t'][0], device="cuda", dtype=torch.float32)
            # 矩阵乘法：X_cam = X_w @ R^T + T
            # verts_cam_m: 此时顶点已被转移至 MICA 物理相机的观察空间中
            verts_cam_m = torch.matmul(verts_flame, R_m.T) + T_m

            # 2. 内参校准 (Intrinsic Alignment)
            K_mica = torch.tensor(payload['opencv']['K'][0], device="cuda", dtype=torch.float32)  # 读取MICA的相机内参
            fx_m, fy_m = K_mica[0, 0], K_mica[1, 1]  # MICA 的焦距
            cx_m, cy_m = K_mica[0, 2], K_mica[1, 2]  # MICA 的焦距

            # 获取 TG 真实图像分辨率
            if hasattr(viewpoint_cam, 'width') and hasattr(viewpoint_cam, 'height'):
                W, H = int(viewpoint_cam.width), int(viewpoint_cam.height)
            else:
                from PIL import Image

                with Image.open(viewpoint_cam.image_path) as _img:
                    W, H = _img.size

            # 根据 TG 相机的视场角 (FoV) 逆向推导 TG 的焦距
            # 公式: $f = \frac{W}{2 \tan(FoV / 2)}$
            fx_tg = W / (2 * math.tan(viewpoint_cam.FoVx / 2.0))
            fy_tg = H / (2 * math.tan(viewpoint_cam.FoVy / 2.0))
            cx_tg, cy_tg = W / 2.0, H / 2.0  # 光心在图像正中央

            # 在相机空间中对齐透视关系
            # 空间畸变补偿（非刚性变形）：
            # 目标：使得点 $P$ 在 TG 内参下的 2D 投影点，严格等于原点在 MICA 内参下的投影点。
            # 根据透视投影方程 $u = \frac{x \cdot f_x}{z} + c_x$ 联立求解得出的代数重构。
            X_adj = verts_cam_m[:, 0] * (fx_m / fx_tg) + verts_cam_m[:, 2] * ((cx_m - cx_tg) / fx_tg)
            Y_adj = verts_cam_m[:, 1] * (fy_m / fy_tg) + verts_cam_m[:, 2] * ((cy_m - cy_tg) / fy_tg)
            Z_adj = verts_cam_m[:, 2]
            # verts_cam_adj: 经过透视补偿后的顶点集。注意：此时三维网格已经发生了沿 Z 轴的非均匀拉伸/裁剪。
            verts_cam_adj = torch.stack([X_adj, Y_adj, Z_adj], dim=1)

            # 3. 场景尺度归一化 (Scale Alignment)
            # 计算 MICA(米) 到 TG(归一化单位) 的绝对空间缩放比例
            C_m = -torch.matmul(R_m.T, T_m)  # MICA 真实相机中心
            dist_m = torch.norm(C_m)  # MICA 场景的相机轨道半径（通常单位为米）
            dist_tg = torch.norm(viewpoint_cam.camera_center)  # TG/COLMAP 场景的相机轨道半径（通常为归一化无量纲单位）
            # 计算空间缩放比例，统一绝对尺度
            scale_factor = dist_tg / dist_m
            verts_cam_tg = verts_cam_adj * scale_factor

            # ========================================================
            # 姿态与位置对齐 (引入三维刚性变换，基于纯净光轴投影)
            # 用当前视角的 Ground Truth 图像特征 (Landmarks) 将三维网格“拉回”正确的位置。
            # ========================================================
            tg_lmks = viewpoint_cam.talking_dict.get('lmks', None)

            if tg_lmks is not None:
                tg_lmks = tg_lmks.to("cuda")

                # ----------------------------------------------------
                # 1. 提取 GT 的绝对解剖学水平线,绝对水平夹角 (双眼连线)
                # 68点中：36-41为一侧眼睛，42-47为另一侧眼睛
                # 利用 68 点 Landmark 中的双眼特征，计算人脸在 2D 图像平面的 Roll (偏航) 旋转角
                # ----------------------------------------------------
                eye_l = tg_lmks[36:42].mean(dim=0)
                eye_r = tg_lmks[42:48].mean(dim=0)

                # 计算双眼连线在 2D 图像平面上的二维向量 (x, y)
                vec_gt_x = eye_r[0] - eye_l[0]
                vec_gt_y = eye_r[1] - eye_l[1]

                # atan2 求取 GT 图像中人脸绝对水平线与图像 X 轴的夹角
                angle_gt = torch.atan2(vec_gt_y, vec_gt_x)

                # ----------------------------------------------------
                # 2. 提取 3D 模型的当前水平方向 (直接读取旋转矩阵，免疫顶点噪声)
                # ----------------------------------------------------
                # 构建 FLAME 标准空间下的绝对水平 X 轴向量
                # v_can_x: FLAME 标准零空间下的 X 轴基底向量 (1, 0, 0)
                # 将标准 X 轴基底旋转至当前的相机物理空间
                v_can_x = torch.tensor([1.0, 0.0, 0.0], device="cuda", dtype=torch.float32)

                # 重新读取 MICA 原始旋转矩阵
                ckpt_path = os.path.join(args.source_path, "checkpoint", f"{int(viewpoint_cam.uid):05d}.frame")
                payload = torch.load(ckpt_path)
                # R_m_orig: 使用 MICA 的原始旋转矩阵
                R_m_orig = torch.tensor(payload['opencv']['R'][0], device="cuda", dtype=torch.float32)

                # 将标准 X 轴转入当前的相机空间
                v_cam_x = torch.matmul(v_can_x, R_m_orig.T)

                # 将相机空间的三维方向向量投影到2D平面上
                vec_proj_x = v_cam_x[0] * fx_tg
                vec_proj_y = v_cam_x[1] * fy_tg

                # 获得 3D 模型当前在2D平面上的投影夹角
                angle_proj = torch.atan2(vec_proj_y, vec_proj_x)

                # 对齐向量方向 (防止向量方向相反导致算出 180 度)
                if torch.cos(angle_gt - angle_proj) < 0:
                    angle_proj += math.pi

                # ----------------------------------------------------
                # 3. 计算并执行绝对精确的 3D 刚性旋转 (Rz)
                # ----------------------------------------------------
                # 误差角 = GT 角度 - 投影角度
                delta_rz = angle_gt - angle_proj

                # 构造标准的绕 Z 轴（此处 Z 轴即相机视线/光轴）的 3D 旋转矩阵
                cos_z = torch.cos(delta_rz)
                sin_z = torch.sin(delta_rz)
                R_z = torch.tensor([
                    [cos_z, -sin_z, 0.0],
                    [sin_z, cos_z, 0.0],
                    [0.0, 0.0, 1.0]
                ], device="cuda", dtype=torch.float32)

                # 围绕 3D 点云的物理质心执行纯净的刚性旋转
                centroid = verts_cam_tg.mean(dim=0)
                verts_cam_tg = torch.matmul(verts_cam_tg - centroid, R_z.T) + centroid

                # ----------------------------------------------------
                # 4. 三维平移校准 (逆透视锚定)
                # ----------------------------------------------------
                # 旋转后重新全量投影
                # 旋转完成后，执行前向透视投影方程 $u = f_x \frac{X}{Z} + c_x$，获取当前的 2D 像素坐标
                u_proj = verts_cam_tg[:, 0] / verts_cam_tg[:, 2] * fx_tg + cx_tg
                v_proj = verts_cam_tg[:, 1] / verts_cam_tg[:, 2] * fy_tg + cy_tg

                # 计算二维图像平面上的平移误差
                # delta_u (水平误差): 使用所有特征点求平均。
                # delta_v (垂直误差): 仅使用下巴顶点 (索引8) 计算极值误差。
                delta_u = tg_lmks[:, 0].mean() - u_proj.mean()
                delta_v = tg_lmks[8, 1] - v_proj.max()
                # 计算系统深度均值
                mean_z = verts_cam_tg[:, 2].mean()

                # 逆透视转换：$\Delta X = \frac{\Delta u \cdot Z}{f_x}$
                # 缺陷指出：正统的透视投影中，每个顶点应乘以自身的 Z 值。
                # 此处统一乘 `mean_z` 是一种“弱透视投影(Weak Perspective)”近似，
                verts_cam_tg[:, 0] += delta_u * mean_z / fx_tg
                verts_cam_tg[:, 1] += delta_v * mean_z / fy_tg

                print(f"[参数补偿] 基于光轴投影校准完成 | Rz 修正: {delta_rz.item() * 180 / math.pi:.2f} 度")
            else:
                print("\n[error] missing lmks")
            # ========================================================
            # TG Camera 坐标系反推回 TG World 坐标系
            # 目的：高斯场 (GaussianModel) 的底层属性 `_xyz` 必须定义在全局的世界坐标系下，以便能够被场景中的所有相机共享和渲染。
            # ========================================================
            # 提取当前 TG 相机到世界坐标系的绝对外参矩阵
            R_tg = torch.tensor(viewpoint_cam.R, device="cuda", dtype=torch.float32)
            T_tg = torch.tensor(viewpoint_cam.T, device="cuda", dtype=torch.float32)

            # 世界坐标反推公式：$X_{world} = (X_{camera} - T) \cdot (R^T)^{-1}$
            init_verts_aligned = torch.matmul(verts_cam_tg - T_tg, torch.inverse(R_tg))
            # 以最终对齐的世界坐标初始化高斯球的叶子张量
            gaussians.create_from_verts(init_verts_aligned)
            # 将新的张量注册进入优化器
            gaussians.training_setup(opt)

            # ================   FLAME初始 (全局仅执行一次)   =====
            if iteration == 1:
                with torch.no_grad():  # 阻断梯度，确保初始化的绝对纯净，防止反向传播污染基底
                    codedict = {}
                    codedict['shape'] = viewpoint_cam.talking_dict['shape']
                    DeformModel.example_init(codedict)

                    codedict['expr'] = viewpoint_cam.talking_dict['exp_param']
                    codedict['eyes_pose'] = viewpoint_cam.talking_dict['eyes_pose']
                    codedict['eyelids'] = viewpoint_cam.talking_dict['eyelids']
                    codedict['jaw_pose'] = viewpoint_cam.talking_dict['jaw_pose']

                    verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

                verts_flame = verts_final[0].detach()  # 剥离计算图，获取纯数值张量

                # ========================================================
                # 空间坐标系绝对对齐 (包含 尺度归一化 与 内参畸变校准)
                # ========================================================

                ckpt_path = os.path.join(args.source_path, "checkpoint", f"{int(viewpoint_cam.uid):05d}.frame")
                payload = torch.load(ckpt_path)

                # 1. MICA World 到 MICA Camera
                # R_m: MICA 的旋转矩阵 (3x3); T_m: MICA 的平移向量 (3,)
                R_m = torch.tensor(payload['opencv']['R'][0], device="cuda", dtype=torch.float32)
                T_m = torch.tensor(payload['opencv']['t'][0], device="cuda", dtype=torch.float32)
                # 矩阵乘法：X_cam = X_w @ R^T + T
                # verts_cam_m: 此时顶点已被转移至 MICA 物理相机的观察空间中
                verts_cam_m = torch.matmul(verts_flame, R_m.T) + T_m

                # 2. 内参校准 (Intrinsic Alignment)
                K_mica = torch.tensor(payload['opencv']['K'][0], device="cuda", dtype=torch.float32)  # 读取MICA的相机内参
                fx_m, fy_m = K_mica[0, 0], K_mica[1, 1]  # MICA 的焦距
                cx_m, cy_m = K_mica[0, 2], K_mica[1, 2]  # MICA 的焦距

                # 获取 TG 真实图像分辨率
                if hasattr(viewpoint_cam, 'width') and hasattr(viewpoint_cam, 'height'):
                    W, H = int(viewpoint_cam.width), int(viewpoint_cam.height)
                else:
                    from PIL import Image

                    with Image.open(viewpoint_cam.image_path) as _img:
                        W, H = _img.size

                # 根据 TG 相机的视场角 (FoV) 逆向推导 TG 的焦距
                # 公式: $f = \frac{W}{2 \tan(FoV / 2)}$
                fx_tg = W / (2 * math.tan(viewpoint_cam.FoVx / 2.0))
                fy_tg = H / (2 * math.tan(viewpoint_cam.FoVy / 2.0))
                cx_tg, cy_tg = W / 2.0, H / 2.0  # 光心在图像正中央

                # 在相机空间中对齐透视关系
                # 空间畸变补偿（非刚性变形）：
                # 目标：使得点 $P$ 在 TG 内参下的 2D 投影点，严格等于原点在 MICA 内参下的投影点。
                # 根据透视投影方程 $u = \frac{x \cdot f_x}{z} + c_x$ 联立求解得出的代数重构。
                X_adj = verts_cam_m[:, 0] * (fx_m / fx_tg) + verts_cam_m[:, 2] * ((cx_m - cx_tg) / fx_tg)
                Y_adj = verts_cam_m[:, 1] * (fy_m / fy_tg) + verts_cam_m[:, 2] * ((cy_m - cy_tg) / fy_tg)
                Z_adj = verts_cam_m[:, 2]
                # verts_cam_adj: 经过透视补偿后的顶点集。注意：此时三维网格已经发生了沿 Z 轴的非均匀拉伸/裁剪。
                verts_cam_adj = torch.stack([X_adj, Y_adj, Z_adj], dim=1)

                # 3. 场景尺度归一化 (Scale Alignment)
                # 计算 MICA(米) 到 TG(归一化单位) 的绝对空间缩放比例
                C_m = -torch.matmul(R_m.T, T_m)  # MICA 真实相机中心
                dist_m = torch.norm(C_m)  # MICA 场景的相机轨道半径（通常单位为米）
                dist_tg = torch.norm(viewpoint_cam.camera_center)  # TG/COLMAP 场景的相机轨道半径（通常为归一化无量纲单位）
                # 计算空间缩放比例，统一绝对尺度
                scale_factor = dist_tg / dist_m
                verts_cam_tg = verts_cam_adj * scale_factor

                # ========================================================
                # 姿态与位置对齐 (引入三维刚性变换，基于纯净光轴投影)
                # 用当前视角的 Ground Truth 图像特征 (Landmarks) 将三维网格“拉回”正确的位置。
                # ========================================================
                tg_lmks = viewpoint_cam.talking_dict.get('lmks', None)

                if tg_lmks is not None:
                    tg_lmks = tg_lmks.to("cuda")

                    # ----------------------------------------------------
                    # 1. 提取 GT 的绝对解剖学水平线,绝对水平夹角 (双眼连线)
                    # 68点中：36-41为一侧眼睛，42-47为另一侧眼睛
                    # 利用 68 点 Landmark 中的双眼特征，计算人脸在 2D 图像平面的 Roll (偏航) 旋转角
                    # ----------------------------------------------------
                    eye_l = tg_lmks[36:42].mean(dim=0)
                    eye_r = tg_lmks[42:48].mean(dim=0)

                    # 计算双眼连线在 2D 图像平面上的二维向量 (x, y)
                    vec_gt_x = eye_r[0] - eye_l[0]
                    vec_gt_y = eye_r[1] - eye_l[1]

                    # atan2 求取 GT 图像中人脸绝对水平线与图像 X 轴的夹角
                    angle_gt = torch.atan2(vec_gt_y, vec_gt_x)

                    # ----------------------------------------------------
                    # 2. 提取 3D 模型的当前水平方向 (直接读取旋转矩阵，免疫顶点噪声)
                    # ----------------------------------------------------
                    # 构建 FLAME 标准空间下的绝对水平 X 轴向量
                    # v_can_x: FLAME 标准零空间下的 X 轴基底向量 (1, 0, 0)
                    # 将标准 X 轴基底旋转至当前的相机物理空间
                    v_can_x = torch.tensor([1.0, 0.0, 0.0], device="cuda", dtype=torch.float32)

                    # 重新读取 MICA 原始旋转矩阵
                    ckpt_path = os.path.join(args.source_path, "checkpoint", f"{int(viewpoint_cam.uid):05d}.frame")
                    payload = torch.load(ckpt_path)
                    # R_m_orig: 使用 MICA 的原始旋转矩阵
                    R_m_orig = torch.tensor(payload['opencv']['R'][0], device="cuda", dtype=torch.float32)

                    # 将标准 X 轴转入当前的相机空间
                    v_cam_x = torch.matmul(v_can_x, R_m_orig.T)

                    # 将相机空间的三维方向向量投影到2D平面上
                    vec_proj_x = v_cam_x[0] * fx_tg
                    vec_proj_y = v_cam_x[1] * fy_tg

                    # 获得 3D 模型当前在2D平面上的投影夹角
                    angle_proj = torch.atan2(vec_proj_y, vec_proj_x)

                    # 对齐向量方向 (防止向量方向相反导致算出 180 度)
                    if torch.cos(angle_gt - angle_proj) < 0:
                        angle_proj += math.pi

                    # ----------------------------------------------------
                    # 3. 计算并执行绝对精确的 3D 刚性旋转 (Rz)
                    # ----------------------------------------------------
                    # 误差角 = GT 角度 - 投影角度
                    delta_rz = angle_gt - angle_proj

                    # 构造标准的绕 Z 轴（此处 Z 轴即相机视线/光轴）的 3D 旋转矩阵
                    cos_z = torch.cos(delta_rz)
                    sin_z = torch.sin(delta_rz)
                    R_z = torch.tensor([
                        [cos_z, -sin_z, 0.0],
                        [sin_z, cos_z, 0.0],
                        [0.0, 0.0, 1.0]
                    ], device="cuda", dtype=torch.float32)

                    # 围绕 3D 点云的物理质心执行纯净的刚性旋转
                    centroid = verts_cam_tg.mean(dim=0)
                    verts_cam_tg = torch.matmul(verts_cam_tg - centroid, R_z.T) + centroid

                    # ----------------------------------------------------
                    # 4. 三维平移校准 (逆透视锚定)
                    # ----------------------------------------------------
                    # 旋转后重新全量投影
                    # 旋转完成后，执行前向透视投影方程 $u = f_x \frac{X}{Z} + c_x$，获取当前的 2D 像素坐标
                    u_proj = verts_cam_tg[:, 0] / verts_cam_tg[:, 2] * fx_tg + cx_tg
                    v_proj = verts_cam_tg[:, 1] / verts_cam_tg[:, 2] * fy_tg + cy_tg

                    # 计算二维图像平面上的平移误差
                    # delta_u (水平误差): 使用所有特征点求平均。
                    # delta_v (垂直误差): 仅使用下巴顶点 (索引8) 计算极值误差。
                    delta_u = tg_lmks[:, 0].mean() - u_proj.mean()
                    delta_v = tg_lmks[8, 1] - v_proj.max()
                    # 计算系统深度均值
                    mean_z = verts_cam_tg[:, 2].mean()

                    # 逆透视转换：$\Delta X = \frac{\Delta u \cdot Z}{f_x}$
                    # 缺陷指出：正统的透视投影中，每个顶点应乘以自身的 Z 值。
                    # 此处统一乘 `mean_z` 是一种“弱透视投影(Weak Perspective)”近似，
                    verts_cam_tg[:, 0] += delta_u * mean_z / fx_tg
                    verts_cam_tg[:, 1] += delta_v * mean_z / fy_tg

                    print(f"[参数补偿] 基于光轴投影校准完成 | Rz 修正: {delta_rz.item() * 180 / math.pi:.2f} 度")
                else:
                    print("\n[error] missing lmks")
                # ========================================================
                # TG Camera 坐标系反推回 TG World 坐标系
                # 目的：高斯场 (GaussianModel) 的底层属性 `_xyz` 必须定义在全局的世界坐标系下，以便能够被场景中的所有相机共享和渲染。
                # ========================================================
                # 提取当前 TG 相机到世界坐标系的绝对外参矩阵
                R_tg = torch.tensor(viewpoint_cam.R, device="cuda", dtype=torch.float32)
                T_tg = torch.tensor(viewpoint_cam.T, device="cuda", dtype=torch.float32)

                # 世界坐标反推公式：$X_{world} = (X_{camera} - T) \cdot (R^T)^{-1}$
                init_verts_aligned = torch.matmul(verts_cam_tg - T_tg, torch.inverse(R_tg))
                # 以最终对齐的世界坐标初始化高斯球的叶子张量
                gaussians.create_from_verts(init_verts_aligned)
                # 将新的张量注册进入优化器
                gaussians.training_setup(opt)





#
        # ================   FLAME初始 (全局仅执行一次)  初始模型初始化 =====
        if iteration == 1:
            with torch.no_grad():
                codedict = {}

                # 1. 构建全零通用基底
                codedict['shape'] = torch.zeros_like(viewpoint_cam.talking_dict['shape'])
                codedict['expr'] = torch.zeros_like(viewpoint_cam.talking_dict['exp_param'])
                codedict['eyelids'] = torch.zeros_like(viewpoint_cam.talking_dict['eyelids'])

                # 2. 姿态参数初始化为 6D Identity
                I_mat = torch.eye(3, device="cuda").unsqueeze(0)
                from pytorch3d.transforms import matrix_to_rotation_6d
                I_6d = matrix_to_rotation_6d(I_mat)

                codedict['jaw_pose'] = I_6d.expand_as(viewpoint_cam.talking_dict['jaw_pose'])
                I_6d_eyes = torch.cat([I_6d, I_6d], dim=1)
                codedict['eyes_pose'] = I_6d_eyes.expand_as(viewpoint_cam.talking_dict['eyes_pose'])

                DeformModel.example_init(codedict)
                verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

            verts_flame = verts_final[0].detach()

            # ========================================================
            # 终极对齐：等比缩放 + 鼻尖绝对锚定 (Nose-Tip Anchoring)
            # ========================================================
            if hasattr(viewpoint_cam, 'width') and hasattr(viewpoint_cam, 'height'):
                W, H = int(viewpoint_cam.width), int(viewpoint_cam.height)
            else:
                from PIL import Image
                with Image.open(viewpoint_cam.image_path) as _img:
                    W, H = _img.size

            fx_tg = W / (2 * math.tan(viewpoint_cam.FoVx / 2.0))
            fy_tg = H / (2 * math.tan(viewpoint_cam.FoVy / 2.0))
            cx_tg, cy_tg = W / 2.0, H / 2.0

            R_tg = torch.tensor(viewpoint_cam.R, device="cuda", dtype=torch.float32)
            T_tg = torch.tensor(viewpoint_cam.T, device="cuda", dtype=torch.float32)

            tg_lmks = viewpoint_cam.talking_dict.get('lmks', None)

            if tg_lmks is not None:
                tg_lmks = tg_lmks.to("cuda")

                verts_cam_tg = verts_flame.clone()

                # 提取 3D 模型中 Z 轴最突出的点作为鼻尖物理锚点
                nose_idx = verts_flame[:, 2].argmax().item()

                # 修复坐标系手性 (Y轴变为向下)
                verts_cam_tg[:, 1] *= -1
                verts_cam_tg[:, 2] *= -1

                base_z = T_tg[2].item()

                # ----------------------------------------------------
                # 锁一：纯等比缩放
                # ----------------------------------------------------
                verts_temp = verts_cam_tg.clone()
                verts_temp[:, 2] += base_z
                u_proj_temp = verts_temp[:, 0] / verts_temp[:, 2] * fx_tg + cx_tg

                proj_w = u_proj_temp.max() - u_proj_temp.min()
                gt_w = tg_lmks[16, 0] - tg_lmks[0, 0]

                # 补偿常数 (2D 脸颊宽 -> 3D 脑壳总宽)
                scale_factor = (gt_w / proj_w) * 1.15

                verts_cam_tg = verts_cam_tg * scale_factor
                verts_cam_tg[:, 2] += base_z

                # ----------------------------------------------------
                # 锁二：全姿态刚性矩阵重构 (Roll + Yaw + Pitch)
                # ----------------------------------------------------
                eye_l_2d = tg_lmks[36:42].mean(dim=0)
                eye_r_2d = tg_lmks[42:48].mean(dim=0)

                # 1. Rz (Roll 翻滚角) - 双眼绝对连线
                vec_gt_x = eye_r_2d[0] - eye_l_2d[0]
                vec_gt_y = eye_r_2d[1] - eye_l_2d[1]
                delta_rz = torch.atan2(vec_gt_y, vec_gt_x)

                # 2. Ry (Yaw 偏航角) - 鼻尖水平视差
                dL = tg_lmks[30, 0] - tg_lmks[0, 0]
                dR = tg_lmks[16, 0] - tg_lmks[30, 0]
                ratio_y = torch.clamp((dL - dR) / (dL + dR), min=-0.99, max=0.99)
                # 负号修正右手坐标系，0.8 为平滑衰减系数防止过激旋转
                delta_ry = -torch.asin(ratio_y) * 0.8

                # 3. Rx (Pitch 俯仰角) - 鼻尖垂直视差
                y_eye = eye_l_2d[1] * 0.5 + eye_r_2d[1] * 0.5
                y_nose = tg_lmks[30, 1]
                y_chin = tg_lmks[8, 1]
                d_top = y_nose - y_eye
                d_bot = y_chin - y_nose
                ratio_x = torch.clamp((d_bot - d_top) / (d_bot + d_top), min=-0.99, max=0.99)
                delta_rx = -torch.asin(ratio_x) * 0.8

                # 构造复合旋转矩阵 R = Rz * Ry * Rx
                cos_z, sin_z = torch.cos(delta_rz), torch.sin(delta_rz)
                R_z = torch.tensor([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], device="cuda",
                                   dtype=torch.float32)

                cos_y, sin_y = torch.cos(delta_ry), torch.sin(delta_ry)
                R_y = torch.tensor([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], device="cuda",
                                   dtype=torch.float32)

                cos_x, sin_x = torch.cos(delta_rx), torch.sin(delta_rx)
                R_x = torch.tensor([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], device="cuda",
                                   dtype=torch.float32)

                R_total = torch.matmul(R_z, torch.matmul(R_y, R_x))

                # 围绕 3D 物理质心执行全姿态旋转，消除钟摆效应
                centroid = verts_cam_tg.mean(dim=0)
                verts_cam_tg = torch.matmul(verts_cam_tg - centroid, R_total.T) + centroid
                # ----------------------------------------------------
                # 锁三：无视差的外轮廓几何中位平移锚定 (核心修复)
                # ----------------------------------------------------
                u_proj = verts_cam_tg[:, 0] / verts_cam_tg[:, 2] * fx_tg + cx_tg
                v_proj = verts_cam_tg[:, 1] / verts_cam_tg[:, 2] * fy_tg + cy_tg

                # 【水平误差】计算 2D 真实人脸轮廓边界的绝对中点
                center_u_2d = (tg_lmks[0, 0] + tg_lmks[16, 0]) / 2.0
                # 【水平误差】计算 3D 投影边框的绝对中点
                center_u_3d = (u_proj.max() + u_proj.min()) / 2.0

                delta_u = center_u_2d - center_u_3d

                # 【垂直误差】维持下巴底线严密扣合 (关键点 8 是面部最低点)
                delta_v = tg_lmks[8, 1] - v_proj.max()

                # 逆透视映射
                verts_cam_tg[:, 0] += delta_u * base_z / fx_tg
                verts_cam_tg[:, 1] += delta_v * base_z / fy_tg

                print(
                    f"\n[系统状态] 终极等比对齐完毕 | 鼻尖锚定成功 | 缩放比: {scale_factor:.3f} | 倾角修正: {delta_rz.item() * 180 / math.pi:.2f}度")
            else:
                print("\n[运行异常] talking_dict 中缺失 lmks。")
                verts_cam_tg = verts_flame.clone()
                verts_cam_tg[:, 1] *= -1
                verts_cam_tg[:, 2] *= -1
                verts_cam_tg[:, 2] += T_tg[2].item()
            # ========================================================

            init_verts_aligned = torch.matmul(verts_cam_tg - T_tg, torch.inverse(R_tg))
            gaussians.create_from_verts(init_verts_aligned)
            gaussians.training_setup(opt)
