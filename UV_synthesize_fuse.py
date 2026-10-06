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
from scene import Scene, GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_motion, render_dual
import torchvision
from utils.general_utils import safe_state
from utils.camera_utils import loadCamOnTheFly
import copy
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.motion_net import MotionNetwork
from src.deform_model import Deform_Model
from networks.dual_branch import DualBranchUNet
from utils.uv_utils import PositionMapGenerator, load_uv_region_masks
from utils.dual_utils import _compute_displacement, _to_axis_angle
from PIL import Image
import torch.nn.functional as F
import math


def render_set_dual(model_path, name, iteration, views, gaussians, motion_net,
                    dual_branch_net, deform_model, map_gen, pipeline, background):
    render_path_dual = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_dual")
    render_path_nodual = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_nodual")
    render_path_diff = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_diff")

    makedirs(render_path_dual, exist_ok=True)
    makedirs(render_path_nodual, exist_ok=True)
    makedirs(render_path_diff, exist_ok=True)

    all_dual = []
    all_nodual = []
    all_diff = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True)):
        if view.original_image is None:
            view = loadCamOnTheFly(copy.deepcopy(view))

        with torch.no_grad():
            td = view.talking_dict
            codedict_frame = {
                'shape': td.get('shape', deform_model.default_shape_code),
                'expr': td.get('expr', deform_model.default_expr_code),
            }
            displacement_map = _compute_displacement(deform_model, map_gen, codedict_frame)
            flame_param = {
                'expr': td.get('expr', torch.zeros(1, 100, device="cuda")).to("cuda"),
                'neck_pose': _to_axis_angle(td.get('neck_pose', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                'jaw_pose': _to_axis_angle(td.get('jaw_pose', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                'eyes_pose': _to_axis_angle(td.get('eyes_pose', torch.zeros(1, 6, device="cuda")), 6).to("cuda"),
                'rotation': _to_axis_angle(td.get('rotation', torch.zeros(1, 3, device="cuda"))).to("cuda"),
                'translation': td.get('translation', torch.zeros(1, 3, device="cuda")).to("cuda"),
            }
            offset_dual = dual_branch_net(flame_param, 0, displacement_map)
            render_dual_pkg = render_dual(view, gaussians, motion_net, offset_dual, pipeline, background)
            render_nodual_pkg = render_motion(view, gaussians, motion_net, pipeline, background, frame_idx=0)

        alpha_d = render_dual_pkg["alpha"]
        img_d = render_dual_pkg["render"] + view.background.cuda() / 255.0 * (1.0 - alpha_d)
        alpha_nd = render_nodual_pkg["alpha"]
        img_nd = render_nodual_pkg["render"] + view.background.cuda() / 255.0 * (1.0 - alpha_nd)

        img_diff = torch.clamp((img_d - img_nd).abs() * 5.0, 0, 1)

        pred_d = (img_d.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        pred_nd = (img_nd.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        pred_diff = (img_diff.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        all_dual.append(pred_d)
        all_nodual.append(pred_nd)
        all_diff.append(pred_diff)

    imageio.mimwrite(os.path.join(render_path_dual, 'out.mp4'), all_dual, fps=25, quality=8, macro_block_size=1)
    imageio.mimwrite(os.path.join(render_path_nodual, 'out.mp4'), all_nodual, fps=25, quality=8, macro_block_size=1)
    imageio.mimwrite(os.path.join(render_path_diff, 'out_diff.mp4'), all_diff, fps=25, quality=8, macro_block_size=1)
    print(f"[Dual] Saved to {render_path_dual}")
    print(f"[NoDual] Saved to {render_path_nodual}")
    print(f"[Diff] Saved to {render_path_diff}")


def render_sets_dual(dataset: ModelParams, iteration: int, pipeline: PipelineParams, use_train: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        motion_net = MotionNetwork(args=dataset).cuda()

        ckpt_path = os.path.join(dataset.model_path, "chkpnt_face_latest.pth")
        ckpt = torch.load(ckpt_path, map_location="cuda")

        model_params = ckpt[0]
        motion_params = ckpt[1]
        gaussians.restore(model_params, None)
        motion_net.load_state_dict(motion_params, strict=False)

        has_dual = len(ckpt) >= 9 and ckpt[6] is not None
        if not has_dual:
            print("[WARN] No DualBranchUNet in checkpoint, falling back to render_motion only")

        if has_dual:
            dual_params = ckpt[6]

            # Deform_Model for per-frame FLAME geometry
            DeformModel = Deform_Model(device="cuda").to("cuda")
            # 初始化 UV 缓存（与训练一致）
            DeformModel.example_init({'shape': torch.zeros(1, 300, device="cuda")})

            # PositionMapGenerator (reso matches training: DeformModel.uv_size = 128)
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

            # UV masks
            uv_mask = load_uv_region_masks("flame/uv_region_masks.pkl")
            for k in uv_mask:
                m = uv_mask[k]
                if m.dim() == 3:
                    m = m.unsqueeze(1)
                if m.shape[-1] != uv_size:
                    m = F.interpolate(m.float(), size=(uv_size, uv_size),
                                      mode='bilinear', align_corners=True)
                uv_mask[k] = (m > 0.5).float()

            # 构建 UV head mask
            uv_head_mask = torch.zeros(1, 1, uv_size * uv_size, device="cuda")
            all_valid_idx = DeformModel.uvmask_flaten_idx.nonzero(as_tuple=False).squeeze(-1)
            head_in_valid = DeformModel.uv_head_idx.nonzero(as_tuple=False).squeeze(-1)
            uv_head_mask[:, :, all_valid_idx[head_in_valid]] = 1.0
            uv_head_mask = uv_head_mask.view(1, 1, uv_size, uv_size)

            # Reference image
            ref_img_path = os.path.join(dataset.source_path, 'gt_imgs', '0.jpg')
            if not os.path.exists(ref_img_path):
                ref_img_path = scene.getTrainCameras()[0].image_path
            reference_image = Image.open(ref_img_path).convert("RGB")

            # UV coords restored from checkpoint (gaussians._uv_coords)
            uv_coords = gaussians.get_uv_coords.unsqueeze(0)

            dual_branch_net = DualBranchUNet(
                device="cuda",
                uv_sample_coords=uv_coords,
                uv_mask=uv_mask,
                reference_image=reference_image,
                position_map=position_map,
                uv_size=uv_size,
                head_mask=uv_head_mask,
            )
            dual_branch_net.load_state_dict(dual_params)
            dual_branch_net.to("cuda").eval()
            print(f"[DualBranchUNet] Loaded, uv_size={uv_size}, uv_coords={uv_coords.shape[1]}")
        else:
            DeformModel = None
            map_gen = None
            dual_branch_net = None

        bg_color = [0, 1, 0]  # matches training
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        cam_list = scene.getTrainCameras() if use_train else scene.getTestCameras()

        if has_dual:
            render_set_dual(dataset.model_path, "test" if not use_train else "train",
                            scene.loaded_iter if iteration == -1 else iteration,
                            cam_list, gaussians, motion_net,
                            dual_branch_net, DeformModel, map_gen, pipeline, background)
        else:
            # Fallback: render without dual branch
            render_path = os.path.join(dataset.model_path, "test", "ours_{}".format(iteration), "renders")
            makedirs(render_path, exist_ok=True)
            all_preds = []
            for idx, view in enumerate(tqdm(cam_list, desc="Rendering", ascii=True)):
                if view.original_image is None:
                    view = loadCamOnTheFly(copy.deepcopy(view))
                with torch.no_grad():
                    render_pkg = render_motion(view, gaussians, motion_net, pipeline, background, frame_idx=0)
                alpha = render_pkg["alpha"]
                image = render_pkg["render"] + view.background.cuda() / 255.0 * (1.0 - alpha)
                pred = (image.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                all_preds.append(pred)
            imageio.mimwrite(os.path.join(render_path, 'out.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--use_train", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets_dual(model.extract(args), args.iteration, pipeline.extract(args), args.use_train)
