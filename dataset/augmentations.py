# Copyright 2020 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image
from dataset.types import is_int, is_seq

def filter_dict(dictionary, keywords):
    """
    Returns only the keywords that are part of a dictionary

    Parameters
    ----------
    dictionary : dict
        Dictionary for filtering
    keywords : list of str
        Keywords that will be filtered

    Returns
    -------
    keywords : list of str
        List containing the keywords that are keys in dictionary
    """
    return [key for key in keywords if key in dictionary]
########################################################################################################################

def parse_crop_borders(borders, shape):
    """
    Calculate borders for cropping.

    Parameters
    ----------
    borders : tuple
        Border input for parsing. Can be one of the following forms:
        (int, int, int, int): y, height, x, width
        (int, int): y, x --> y, height = image_height - y, x, width = image_width - x
        Negative numbers are taken from image borders, according to the shape argument
        Float numbers for y and x are treated as percentage, according to the shape argument,
            and in this case height and width are centered at that point.
    shape : tuple
        Image shape (image_height, image_width), used to determine negative crop boundaries

    Returns
    -------
    borders : tuple (left, top, right, bottom)
        Parsed borders for cropping
    """
    if len(borders) == 0:
        return 0, 0, shape[1], shape[0]
    borders = list(borders).copy()
    if len(borders) == 4:
        borders = [borders[2], borders[0], borders[3], borders[1]]
        if is_int(borders[0]):
            borders[0] += shape[1] if borders[0] < 0 else 0
            borders[2] += shape[1] if borders[2] <= 0 else borders[0]
        else:
            center_w, half_w = borders[0] * shape[1], borders[2] / 2
            borders[0] = int(center_w - half_w)
            borders[2] = int(center_w + half_w)
        if is_int(borders[1]):
            borders[1] += shape[0] if borders[1] < 0 else 0
            borders[3] += shape[0] if borders[3] <= 0 else borders[1]
        else:
            center_h, half_h = borders[1] * shape[0], borders[3] / 2
            borders[1] = int(center_h - half_h)
            borders[3] = int(center_h + half_h)
    elif len(borders) == 2:
        borders = [borders[1], borders[0]]
        if is_int(borders[0]):
            borders = (max(0, borders[0]),
                       max(0, borders[1]),
                       shape[1] + min(0, borders[0]),
                       shape[0] + min(0, borders[1]))
        else:
            center_w, half_w = borders[0] * shape[1], borders[1] / 2
            center_h, half_h = borders[0] * shape[0], borders[1] / 2
            borders = (int(center_w - half_w), int(center_h - half_h),
                       int(center_w + half_w), int(center_h + half_h))
    else:
        raise NotImplementedError('Crop tuple must have 2 or 4 values.')
    assert 0 <= borders[0] < borders[2] <= shape[1] and \
           0 <= borders[1] < borders[3] <= shape[0], 'Crop borders {} are invalid'.format(borders)
    return borders

def random_crop_borders(shape, scale=(0.6,1.0), ratio=(0.75,1.33)):
    """
    Calculate borders for cropping.

    Parameters
    ----------
    shape : tuple
        Image shape (image_height, image_width), used to determine negative crop boundaries
    scale (tuple of python:float, optional)  Specifies the lower and upper bounds for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image.

    ratio (tuple of python:float, optional)  lower and upper bounds for the random aspect ratio of the crop, before resizing.
    Returns
    -------
    borders : tuple (left, top, right, bottom)
        Parsed borders for cropping
    """
    image_height, image_width = shape

    scale_factor = random.uniform(scale[0], scale[1])
    ratio_factor = random.uniform(ratio[0], ratio[1])

    area = image_height * image_width
    target_area = scale_factor * area

    image_ratio = image_width / image_height

    w = int(round((target_area * ratio_factor * image_ratio) ** 0.5))
    h = int(round((target_area / ratio_factor / image_ratio) ** 0.5))

    w = min(w, image_width)
    h = min(h, image_height)

    x = random.randint(0, image_width - w)
    y = random.randint(0, image_height - h)

    borders = (x, y, x + w, y + h)
    
    # Return updated borders
    return borders

def resize_image(image, shape, interpolation=Image.LANCZOS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(depth, axis=2)


def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    if depth is None:
        return depth
    if not is_seq(shape):
        shape = tuple(int(s * shape) for s in depth.shape)
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    idx = x > 0
    crd, val = uv[idx], x[idx]
    crd[:, 0] = (crd[:, 0] * (shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w)).astype(np.int32)
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    return np.expand_dims(depth, axis=2)


def resize_sample_image_and_intrinsics(sample, shape,
                                       image_interpolation=Image.LANCZOS):
    """
    Resizes the image and intrinsics of a sample

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample['rgb'].size
    (out_h, out_w) = shape
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        intrinsics = np.copy(sample[key])
        intrinsics[0] *= out_w / orig_w
        intrinsics[1] *= out_h / orig_h
        sample[key] = intrinsics
    for key in filter_dict(sample, [
        'rgb',  'mask', # 'rgb_original',
    ]):
        sample[key] = image_transform(sample[key])
    for key in filter_dict(sample, [
        'rgb_context', # 'rgb_context_original',
    ]):
        sample[key] = [image_transform(k) for k in sample[key]]
    return sample

def resize_sample(sample, shape, image_interpolation=Image.LANCZOS):
    """
    Resizes a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)
    for key in filter_dict(sample, [
        'gt_depth', 'input_depth',
    ]):
        sample[key] = resize_depth_preserve(sample[key], shape)
    for key in filter_dict(sample, [
        'depth_context',
    ]):
        sample[key] = [resize_depth_preserve(k, shape) for k in sample[key]]
    return sample

########################################################################################################################

def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth', 'input_depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    return sample

########################################################################################################################

def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.

    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue, color)
        Color jittering parameters
    prob : float
        Jittering probability

    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        color_jitter_transform = random_color_jitter_transform(parameters[:4])
        if len(parameters) > 4 and parameters[4] > 0:
            matrix = (random.uniform(1. - parameters[4], 1 + parameters[4]), 0, 0, 0,
                      0, random.uniform(1. - parameters[4], 1 + parameters[4]), 0, 0,
                      0, 0, random.uniform(1. - parameters[4], 1 + parameters[4]), 0)
        else:
            matrix = None
        for key in filter_dict(sample, [
            'rgb'
        ]):
            sample[key] = color_jitter_transform(sample[key])
            if matrix is not None:  # If applying color transformation
                sample[key] = sample[key].convert('RGB', matrix)
        for key in filter_dict(sample, [
            'rgb_context'
        ]):
            sample[key] = [color_jitter_transform(k) for k in sample[key]]
            if matrix is not None:  # If applying color transformation
                sample[key] = [k.convert('RGB', matrix) for k in sample[key]]
    # Return jittered (?) sample
    return sample


def random_color_jitter_transform(parameters):
    """
    Creates a reusable color jitter transformation

    Parameters
    ----------
    parameters : tuple (brightness, contrast, saturation, hue, color)
        Color jittering parameters

    Returns
    -------
    transform : torch.vision.Transform
        Color jitter transformation with fixed parameters
    """
    brightness, contrast, saturation, hue = parameters
    brightness = [max(0, 1 - brightness), 1 + brightness]
    contrast = [max(0, 1 - contrast), 1 + contrast]
    saturation = [max(0, 1 - saturation), 1 + saturation]
    hue = [-hue, hue]

    all_transforms = []

    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_brightness(img, brightness_factor)))
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_contrast(img, contrast_factor)))
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_saturation(img, saturation_factor)))
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_hue(img, hue_factor)))
    random.shuffle(all_transforms)
    return transforms.Compose(all_transforms)


def crop_image(image, borders):
    """
    Crop a PIL Image

    Parameters
    ----------
    image : PIL.Image
        Input image
    borders : tuple (left, top, right, bottom)
        Borders used for cropping

    Returns
    -------
    image : PIL.Image
        Cropped image
    """
    return image.crop(borders)


def crop_intrinsics(intrinsics, borders):
    """
    Crop camera intrinsics matrix

    Parameters
    ----------
    intrinsics : np.array [3,3]
        Original intrinsics matrix
    borders : tuple
        Borders used for cropping (left, top, right, bottom)
    Returns
    -------
    intrinsics : np.array [3,3]
        Cropped intrinsics matrix
    """
    intrinsics = np.copy(intrinsics)
    intrinsics[0, 2] -= borders[0]
    intrinsics[1, 2] -= borders[1]
    return intrinsics


def crop_depth(depth, borders):
    """
    Crop a numpy depth map

    Parameters
    ----------
    depth : np.array
        Input numpy array
    borders : tuple
        Borders used for cropping (left, top, right, bottom)

    Returns
    -------
    image : np.array
        Cropped numpy array
    """
    if depth is None:
        return depth
    return depth[borders[1]:borders[3], borders[0]:borders[2]]


def crop_sample_input(sample, borders):
    """
    Crops the input information of a sample (i.e. that go to the networks)

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    borders : tuple
        Borders used for cropping (left, top, right, bottom)

    Returns
    -------
    sample : dict
        Cropped sample
    """
    for key in filter_dict(sample, [
        'intrinsics'
    ]):
        if key + '_full' not in sample.keys():
            sample[key + '_full'] = np.copy(sample[key])
        sample[key] = crop_intrinsics(sample[key], borders)
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'warped_rgb',
    ]):
        sample[key] = crop_image(sample[key], borders)
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original',
    ]):
        sample[key] = [crop_image(val, borders) for val in sample[key]]
    for key in filter_dict(sample, [
        'input_depth', 'bbox2d_depth', 'bbox3d_depth'
    ]):
        sample[key] = crop_depth(sample[key], borders)
    for key in filter_dict(sample, [
        'input_depth_context',
    ]):
        sample[key] = [crop_depth(val, borders) for val in sample[key]]
    return sample


def crop_sample_supervision(sample, borders):
    """
    Crops the output information of a sample (i.e. ground-truth supervision)

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    borders : tuple
        Borders used for cropping

    Returns
    -------
    sample : dict
        Cropped sample
    """
    for key in filter_dict(sample, [
        'depth', 'bbox2d_depth', 'bbox3d_depth', 'semantic',
        'bwd_optical_flow', 'fwd_optical_flow', 'valid_fwd_optical_flow',
        'bwd_scene_flow', 'fwd_scene_flow', 'mask'
    ]):
        sample[key] = crop_depth(sample[key], borders)
    for key in filter_dict(sample, [
        'depth_context', 'semantic_context',
        'bwd_optical_flow_context', 'fwd_optical_flow_context',
        'bwd_scene_flow_context', 'fwd_scene_flow_context',
    ]):
        sample[key] = [crop_depth(k, borders) for k in sample[key]]
    return sample


def crop_sample(sample, borders, prob=1.0):
    """
    Crops a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    borders : tuple
        Borders used for cropping

    Returns
    -------
    sample : dict
        Cropped sample
    """
    if random.random() < prob:
        sample = crop_sample_input(sample, borders)
        sample = crop_sample_supervision(sample, borders)
    return sample