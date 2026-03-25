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

import imageio
import numpy as np
import torch
from scene import Scene
import os
import copy
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_motion, render_motion_mouth_con
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, MotionNetwork, MouthMotionNetwork

import torch.nn.functional as F


def dilate_fn(bin_img, ksize=13):
    pad = (ksize - 1) // 2
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=pad)
    return out

def render_set(model_path, name, iteration, views, gaussians, motion_net, pipeline, background, fast, dilate,
               personalized):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    all_preds = []
    all_gts = []

    all_preds_face = []

    # === [FPS 测试初始化] ===
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    warmup_frames = 5
    # ========================

    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True)):
        with torch.no_grad():
            # === [开始计时] ===
            starter.record()

            render_pkg = render_motion(view, gaussians, motion_net, pipeline, background, frame_idx=0,
                                       personalized=personalized, align=True)

        alpha = render_pkg["alpha"]
        # 背景融合属于 GPU 渲染计算的一部分，纳入计时域
        image = render_pkg["render"] + view.background.cuda() / 255.0 * (1.0 - alpha)

        # === [结束计时] 必须在 .cpu() 内存拷贝之前执行 ===
        ender.record()
        torch.cuda.synchronize()
        if idx >= warmup_frames:
            timings.append(starter.elapsed_time(ender))
        # ===============================================

        # 以下数据回传与格式转换属于 I/O 准备，不计入模型推理性能
        pred = (image[0:3, ...].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        all_preds.append(pred)

        if not fast:
            all_preds_face.append(
                (render_pkg["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
            # all_preds_mouth.append((render_pkg_mouth["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()* 255).astype(np.uint8))

            all_gts.append(view.original_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))

    # === [FPS 数据计算与输出] ===
    if len(timings) > 0:
        avg_latency = np.mean(timings)
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        print(f"\n--- Performance Metrics ---")
        print(f"Evaluated Frames : {len(timings)} (excluded {warmup_frames} warm-up frames)")
        print(f"Pure GPU Latency : {avg_latency:.2f} ms")
        print(f"Pure GPU FPS     : {fps:.2f}")
        print(f"---------------------------\n")
    # ============================

    imageio.mimwrite(os.path.join(render_path, 'kp_oba.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
    if not fast:
        imageio.mimwrite(os.path.join(gts_path, 'kp_oba.mp4'), all_gts, fps=25, quality=8, macro_block_size=1)

        imageio.mimwrite(os.path.join(render_path, 'out_face.mp4'), all_preds_face, fps=25, quality=8,
                         macro_block_size=1)

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, use_train: bool, fast, dilate,
                personalized):
    with torch.no_grad():
        dataset.type = "face"
        gaussians = GaussianModel(copy.deepcopy(dataset))
        scene = Scene(dataset, gaussians, shuffle=False)

        motion_net = MotionNetwork(args=dataset).cuda()
        (model_params, motion_params, _, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians.restore(model_params, None)


        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # --- 严谨处理输出目录名称 ---
        # 1. 确定基础名称
        base_name = "train" if use_train else "test"

        # 2. 尝试获取 audio_file_path。此处假设它可能被存在 dataset 或 args 中。
        # 使用 getattr 进行安全获取，避免 AttributeError
        audio_path = getattr(dataset, 'audio_file_path', None)

        # 3. 如果存在路径，严谨剥离目录和后缀名
        if audio_path:
            # os.path.basename 去除路径，os.path.splitext 分离文件名和后缀
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_name = f"{base_name}_{audio_name}"
        else:
            output_name = base_name
        # ----------------------------

        render_set(dataset.model_path, output_name, scene.loaded_iter,
                   scene.getTestCameras() if not use_train else scene.getTrainCameras(), gaussians, motion_net,
                   pipeline, background, fast, dilate, personalized)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dilate", action="store_true")
    parser.add_argument("--personalized", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.use_train, args.fast, args.dilate,
                args.personalized)