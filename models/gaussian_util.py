#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import torch
from torch import Tensor
from jaxtyping import Float
from math import isqrt, tan, atan
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum

import numpy as np

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(novel_FovX, 
           novel_FovY, 
           novel_height, 
           novel_width, 
           novel_world_view_transform, 
           novel_full_proj_transform, 
           novel_camera_center, 
           pts_xyz, 
           pts_rgb, 
           rotations, 
           scales, 
           opacity, 
           shs, 
           sh_degree,
           bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32).cuda()
 
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True).cuda()
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = tan(novel_FovX * 0.5)
    tanfovy = tan(novel_FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(novel_height),
        image_width=int(novel_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=novel_world_view_transform,
        projmatrix=novel_full_proj_transform,
        sh_degree=sh_degree,
        campos=novel_camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, _, _ = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=pts_rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)


    return rendered_image

def depth2pc(depth, extrinsic, intrinsic):
    if len(depth.shape) == 4:
        depth = depth.squeeze(1)

    B, H, W = depth.shape
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, H-0.5, H, device=depth.device), torch.linspace(0.5, W-0.5, W, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B H W 3

    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)

    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)

def pc2depth(points, extrinsic, intrinsic, height, width):
    """
    Project 3D points back to depth map (reverse of depth2pc)
    
    Args:
        points: [B, N, 3] - 3D points in world/ego coordinate
        extrinsic: [B, 4, 4] - camera extrinsic matrix (e2c: ego to camera)
        intrinsic: [B, 4, 4] - camera intrinsic matrix
        height: int - target depth map height
        width: int - target depth map width
        
    Returns:
        depth: [B, H, W] - projected depth map
    """
    B, N, _ = points.shape
    device = points.device
    
    # Transform points to camera coordinate
    rot = extrinsic[:, :3, :3]  # [B, 3, 3]
    trans = extrinsic[:, :3, 3:]  # [B, 3, 1]
    
    # points: [B, N, 3] -> [B, 3, N]
    pts_world = points.permute(0, 2, 1)  # [B, 3, N]
    
    # Transform to camera coordinate: pts_cam = R @ pts_world + t
    pts_cam = torch.bmm(rot, pts_world) + trans  # [B, 3, N]
    
    # Project to image plane
    # Normalize by depth (z-coordinate)
    pts_cam_xy = pts_cam[:, :2, :] / (pts_cam[:, 2:3, :] + 1e-8)  # [B, 2, N]
    
    # Apply intrinsic transformation
    fx = intrinsic[:, 0, 0][:, None]  # [B, 1]
    fy = intrinsic[:, 1, 1][:, None]  # [B, 1]
    cx = intrinsic[:, 0, 2][:, None]  # [B, 1]
    cy = intrinsic[:, 1, 2][:, None]  # [B, 1]
    
    u = pts_cam_xy[:, 0, :] * fx + cx  # [B, N]
    v = pts_cam_xy[:, 1, :] * fy + cy  # [B, N]
    depth_values = pts_cam[:, 2, :]  # [B, N]
    
    # Initialize depth map
    depth_maps = torch.zeros(B, height, width, device=device, dtype=points.dtype)
    
    # Project points to pixel coordinates
    for b in range(B):
        # Filter valid points (within image bounds and positive depth)
        valid_mask = (
            (u[b] >= 0) & (u[b] < width) &
            (v[b] >= 0) & (v[b] < height) &
            (depth_values[b] > 0)
        )
        
        if valid_mask.sum() > 0:
            valid_u = u[b][valid_mask].round().long()
            valid_v = v[b][valid_mask].round().long()
            valid_depth = depth_values[b][valid_mask]
            
            # Additional bounds check after rounding to prevent out-of-bounds access
            final_mask = (
                (valid_u >= 0) & (valid_u < width) &
                (valid_v >= 0) & (valid_v < height)
            )
            
            if final_mask.sum() > 0:
                final_u = valid_u[final_mask]
                final_v = valid_v[final_mask]
                final_depth = valid_depth[final_mask]
                
                # Handle multiple points projecting to the same pixel (take closest depth)
                # Create a temporary depth map for this batch
                temp_depth = torch.full((height, width), float('inf'), device=device, dtype=points.dtype)
                temp_depth[final_v, final_u] = torch.minimum(temp_depth[final_v, final_u], final_depth)
                
                # Copy non-inf values to final depth map
                finite_mask = torch.isfinite(temp_depth)
                depth_maps[b][finite_mask] = temp_depth[finite_mask]
    
    return depth_maps

def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    for degree in range(isqrt(n)):
        sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (Hamilton convention: q = [w, x, y, z])
    q1, q2: [..., 4]
    Returns: q = q1 * q2
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = np.split(quaternion0, 4, axis=-1)
    w1, x1, y1, z1 = np.split(quaternion1, 4, axis=-1)
    return np.concatenate((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), axis=-1)
    
def focal2fov(focal, pixels):
    return 2*atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)