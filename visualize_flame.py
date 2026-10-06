"""可视化 FLAME 3D 重建： GT | FLAME mesh(光影+顶点+描边) | 叠图 """
import os, math, json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex,
)
from src.deform_model import Deform_Model
from utils.general_utils import safe_state, load_binary_pickle
from utils.dual_utils import _to_6d


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_path", type=str, required=True)
    parser.add_argument("-m", "--model_path", type=str, default="output/flame_vis")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.model_path, exist_ok=True)
    safe_state(True)

    with open(os.path.join(args.source_path, "transforms_train.json")) as f:
        data = json.load(f)
    focal = data.get("focal_len", 1200)
    frames = data["frames"]

    DeformModel = Deform_Model(device="cuda").to("cuda")

    # 顶点语义分类（柔色）
    masks = load_binary_pickle("flame/FLAME_masks/FLAME_masks.pkl")
    part = np.zeros(5023, dtype=np.int8)
    for r, pid in [('eye_region', 3), ('nose', 4), ('lips', 5), ('forehead', 6)]:
        for v in masks.get(r, []): part[v] = pid
    for v in DeformModel.neckhead_id_list:
        if part[v] == 0: part[v] = 1
    # 柔色：低饱和
    pal = np.array([
        [0.6, 0.58, 0.56],   # 0 other 暖灰
        [0.85, 0.78, 0.65],  # 1 head  肤色
        [0.0, 0.0, 0.0],     # 2 neck (unused)
        [0.7, 0.55, 0.55],   # 3 eye
        [0.72, 0.6, 0.52],   # 4 nose
        [0.75, 0.5, 0.55],   # 5 lips
        [0.62, 0.65, 0.68],  # 6 forehead
    ], dtype=np.float32)
    vert_colors = torch.from_numpy(pal[part]).cuda()

    # 脸部三角面索引
    face_idx = DeformModel.flame_model.faces.long().cpu().numpy()

    frames_dir = os.path.join(args.model_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    n = min(args.frames, len(frames))

    for idx in tqdm(range(n), desc="Rendering FLAME"):
        frame = frames[idx]
        idx_str = f"{idx:05d}"
        img_path = os.path.join(args.source_path, 'gt_imgs', f"{idx}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(args.source_path, 'gt_imgs', idx_str + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt = np.array(Image.open(img_path).convert("RGB"))
        W, H = gt.shape[1], gt.shape[0]

        c2w = np.array(frame["transform_matrix"])
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T.astype(np.float32)
        T = w2c[:3, 3].astype(np.float32)
        R_tg = torch.from_numpy(R).cuda()
        T_tg = torch.from_numpy(T).cuda()

        npz_path = os.path.join(args.source_path, 'flame_param', idx_str + ".npz")
        if not os.path.exists(npz_path): continue
        td = dict(np.load(npz_path, allow_pickle=True))

        def _e(t, dim):
            if t is None: return torch.zeros(1, dim, device="cuda")
            t = torch.as_tensor(t, device="cuda").float()
            return t.unsqueeze(0) if t.dim() == 1 else t

        with torch.no_grad():
            flame_geo = DeformModel.flame_model.forward_geo(
                shape_params=_e(td.get('shape'), 300),
                expression_params=_e(td.get('expr'), 100),
                jaw_pose_params=_to_6d(_e(td.get('jaw_pose'), 6)) if td.get('jaw_pose') is not None else None,
                eye_pose_params=_to_6d(_e(td.get('eyes_pose'), 12), 12) if td.get('eyes_pose') is not None else None,
                rot_params=_to_6d(_e(td.get('rotation'), 3)) if td.get('rotation') is not None else None,
                neck_pose_params=_to_6d(_e(td.get('neck_pose'), 3)) if td.get('neck_pose') is not None else None,
                trans_params=_e(td.get('translation'), 3) if td.get('translation') is not None else None,
            )
            verts_world = flame_geo[0]

            # 对齐（同训练）
            va = verts_world.clone()
            va[:, 1] *= -1
            va[:, 2] *= -1
            va[:, 2] += T_tg[2].item()
            verts_w = torch.matmul(va - T_tg, torch.inverse(R_tg))

            # pytorch3d 渲染（带光影）
            fov_deg = 2 * math.atan(W / (2 * focal)) * 180.0 / math.pi
            cameras = FoVPerspectiveCameras(
                device="cuda", R=R_tg.T[None], T=T_tg[None],
                znear=0.01, zfar=100, fov=fov_deg, degrees=True)
            lights = PointLights(
                device="cuda", location=[[0, 0.5, 2]],
                ambient_color=[[0.5, 0.5, 0.5]],
                diffuse_color=[[0.7, 0.7, 0.7]],
                specular_color=[[0.2, 0.2, 0.2]])
            raster = RasterizationSettings(
                image_size=(H, W), blur_radius=0.0,
                faces_per_pixel=2, perspective_correct=True)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
                shader=SoftPhongShader(device="cuda", cameras=cameras, lights=lights),
            ).to("cuda")

            mesh = Meshes(
                verts=verts_w.unsqueeze(0),
                faces=torch.from_numpy(face_idx).long().to("cuda").unsqueeze(0),
                textures=TexturesVertex(verts_features=vert_colors.unsqueeze(0)),
            )
            flame_img = renderer(mesh)[0, ..., :3]      # [H, W, 3] pytorch3d: y=0 bottom
            flame_img = torch.flip(flame_img, [0])       # 翻到 y=0 top
            flame_img = torch.flip(flame_img, [1])       # 左右翻正
            flame_np = (flame_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        alpha = 0.5
        overlay = (gt.astype(np.float32) * (1 - alpha) + flame_np.astype(np.float32) * alpha).astype(np.uint8)
        comparison = np.concatenate([gt, flame_np, overlay], axis=1)
        Image.fromarray(comparison).save(os.path.join(frames_dir, idx_str + ".png"))

    print(f"Saved {n} frames to {frames_dir}")


if __name__ == "__main__":
    main()
