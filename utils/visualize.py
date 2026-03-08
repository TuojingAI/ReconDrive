import matplotlib.pyplot as plt

import torch

_DEGTORAD = 0.0174533
        

def aug_depth_params(K, n_steps= 75):
    """
    This function augments camera parameters for depth synthesis.
    """
    # augmented parameters for visualization
    aug_params = []
    
    # roll augmentations
    roll_aug = [i for i in range(0, n_steps + 1, 2)] + [i for i in range(n_steps, -n_steps - 1, -2)] + [i for i in range(-n_steps, 1, 2)]
    ang_y, ang_z = 0.0, 0.0
    for angle in roll_aug:
        ang_x = _DEGTORAD * (angle / n_steps * 10.)     
        aug_params.append([torch.inverse(K), ang_x, ang_y, ang_z])        

    # pitch augmentations
    pitch_aug = [i for i in range(0, 50 + 1, 2)] + [i for i in range(50, -50 - 1, -2)] + [i for i in range(-50, 1, 2)]
    ang_x, ang_z = 0.0, 0.0
    for angle in pitch_aug:
        ang_y = _DEGTORAD * (angle / 10.)                
        aug_params.append([torch.inverse(K), ang_x, ang_y, ang_z])
        
    # focal length augmentations
    focal_ratio = K[:, 1, 0, 0] / K[:, 0, 0, 0]
    focal_ratio_aug = focal_ratio / 1.5
    ang_x, ang_y, ang_z = 0.0, 0.0, 0.0
     
    for f_idx in range(100 + 1):
        f_scale = (f_idx / 100. * focal_ratio_aug + (1 - f_idx / 100.))[:, None]
        K_aug = K.clone()
        K_aug[:, :, 0, 0] *= f_scale
        K_aug[:, :, 1, 1] *= f_scale
        aug_params.append([torch.inverse(K_aug), ang_x, ang_y, ang_z])

    for f_idx in range(50 + 1):
        f_scale = (f_idx / 50. * focal_ratio + (1 - f_idx / 50.) * focal_ratio_aug)[:, None]
        K_aug = K.clone()
        K_aug[:, :, 0, 0] *= f_scale
        K_aug[:, :, 1, 1] *= f_scale
        aug_params.append([torch.inverse(K_aug), ang_x, ang_y, ang_z])

    # yaw augmentations
    yaw_aug = [i for i in range(360)]
    inv_K_aug = torch.inverse(K_aug)
    ang_x, ang_y = 0.0, 0.0
    for i in yaw_aug:
        ratio_i = i / 360.
        ang_z = _DEGTORAD * 360 * ratio_i
        aug_params.append([inv_K_aug, ang_x, ang_y, ang_z])
    return aug_params
    
    
def colormap(vis, normalize=True, torch_transpose=True):
    """
    This function visualizes disparity map using colormap specified with disparity map variable.
    """
    disparity_map = plt.get_cmap('plasma', 256)  # for plotting

    if isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()

    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d
        
    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = disparity_map(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = disparity_map(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = disparity_map(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)
    return vis        