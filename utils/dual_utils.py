import torch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d


def _build_uv_coords_from_grid(uvmask_flaten_idx, uv_head_idx, uv_size, device, filter_head=True):
    """从光栅化网格构建每个高斯的 UV 坐标 [1, N, 2]"""
    row = torch.arange(uv_size, device=device).float()
    col = torch.arange(uv_size, device=device).float()
    grid_v, grid_u = torch.meshgrid(row, col, indexing='ij')
    grid_u = grid_u.flatten()
    grid_v = grid_v.flatten()

    valid_u = (grid_u[uvmask_flaten_idx] + 0.5) / uv_size
    valid_v = (grid_v[uvmask_flaten_idx] + 0.5) / uv_size
    if filter_head:
        valid_u = valid_u[uv_head_idx]
        valid_v = valid_v[uv_head_idx]

    return torch.stack([valid_u, valid_v], dim=-1).unsqueeze(0)


def _compute_displacement(deform_model, map_gen, codedict):
    """根据 codedict 计算 UV 位移图。若关键参数缺失返回零图。"""
    td = codedict
    has_expr = 'expr' in td and td['expr'] is not None
    has_jaw = 'jaw_pose' in td and td['jaw_pose'] is not None
    if not has_expr and not has_jaw:
        return torch.zeros(1, 3, deform_model.uv_size, deform_model.uv_size, device=deform_model.device)

    shape_code = td.get('shape', deform_model.default_shape_code)
    expr_code = td.get('expr', deform_model.default_expr_code)
    jaw_pose = td.get('jaw_pose', None)
    eyes_pose = td.get('eyes_pose', None)
    eyelids = td.get('eyelids', None)

    def _to_batch(t, device):
        if t is None:
            return None
        t = t.detach().to(device).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    shape_code = _to_batch(shape_code, deform_model.device)
    expr_code = _to_batch(expr_code, deform_model.device)
    jaw_pose = _to_batch(jaw_pose, deform_model.device)
    eyes_pose = _to_batch(eyes_pose, deform_model.device)
    eyelids = _to_batch(eyelids, deform_model.device)

    geometry = deform_model.flame_model.forward_geo(
        shape_params=shape_code,
        expression_params=expr_code,
        jaw_pose_params=jaw_pose,
        eye_pose_params=eyes_pose,
        eyelid_params=eyelids,
    )
    return map_gen.displacement_map(geometry)


def _to_axis_angle(t, expected_dim=3):
    """6D rotation → axis-angle，适配 DualBranchUNet 的 3D 输入"""
    if t is None:
        return None
    if t.shape[-1] == expected_dim:
        return t
    if t.shape[-1] == 6:
        return matrix_to_axis_angle(rotation_6d_to_matrix(t.unsqueeze(0))).squeeze(0)
    if t.shape[-1] == 12:  # eyes: 2×6D → 2×3
        return matrix_to_axis_angle(rotation_6d_to_matrix(t.view(-1, 6))).view(6)
    return t


def _to_6d(t, expected_6d_dim=6):
    """axis-angle → 6D rotation，适配 DeformModel 的 6D 输入（与 _to_axis_angle 反向）"""
    if t is None:
        return None
    if t.shape[-1] == expected_6d_dim:
        return t
    if t.shape[-1] == 3:  # single joint axis-angle → 6D
        return matrix_to_rotation_6d(axis_angle_to_matrix(t.unsqueeze(0))).squeeze(0)
    if t.shape[-1] == 6:  # eyes: 2×axis-angle → 2×6D
        return matrix_to_rotation_6d(axis_angle_to_matrix(t.view(-1, 3))).view(-1, 12)
    return t
