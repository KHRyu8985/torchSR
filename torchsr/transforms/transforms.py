import functools
import math
import numbers
from typing import List, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F

from PIL import ImageOps

#import albumentations.functional as AF   ### SHKIM albumentations
import albumentations   ### SHKIM albumentations

__all__ = ('ToTensor', 'ToPILImage', 'Compose',
           'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomFlipTurn',
           'RandomCrop_shift', 'RandomCrop_rotate', 'RandomCrop_distort', 'RandomCrop', 'CenterCrop', 'AdjustToScale',
           'ColorJitter', 'GaussianBlur', 'Resize', 'EdgeRemove', 'Crop',
#           'AlbuToTensor', 'AlbuCompose',
#           'AlbuRandomHorizontalFlip', 'AlbuRandomVerticalFlip', 'AlbuRandomFlipTurn',
#           'AlbuRandomCrop_shift', 'AlbuRandomCrop_rotate', 'AlbuRandomCrop_distort', 'AlbuRandomCrop', 'AlbuCenterCrop', 'AlbuAdjustToScale',
            'AlbuRandomCrop_distort', 'AlbuRandomCrop_shift',)
#           'AlbuColorJitter', 'AlbuGaussianBlur')

class BarrelDeformer:
    def __init__(self, w, h, k_1):
        self.w, self.h = w, h
        self.k_1 = 0.01 * (48/min(w,h)) * k_1
#        self.k_2 = 0.01
#        self.k_3 = 0.001
#        self.k_4 = 0.0001

    def transform(self, x, y):
        x_c, y_c = self.w / 2, self.h /2
        x = (x - x_c) / (x_c)
        y = (y - y_c) / (y_c)
        p = 1
        radius = np.sqrt((x*p)**2 + (y*p)**2)
#        m_r = radius**2
        m_r = 1 + self.k_1*radius
#        m_r = 1 + self.k_1*radius + self.k_2*radius**2
#        m_r = 1 + self.k_1*radius + self.k_2*radius**2 + self.k_3*radius**3 + self.k_4*radius**4
        x, y = x * m_r, y * m_r
        x, y = x * x_c + x_c, y * y_c + y_c

#        if x < 0:
#            x = 0
#        if y < 0:
#            y = 0
#        if x > self.w:
#            x = self.w-1
#        if y > self.h:
#            y = self.h-1
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        gridspace = 10
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, min(self.w - 1, x + gridspace), min(self.h - 1, y + gridspace)))
                #target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]


def apply_all(x, func):
    """
    Apply a function to a list of tensors/images or a single tensor/image
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [func(t) for t in x]
    else:
        return func(x)


def remove_numpy(x):
    """
    Transform numpy arrays to Pil Images, so we can apply torchvision transforms
    """
    if isinstance(x, np.ndarray):
        return PIL.Image.fromarray(x)
    return x


def smallest_image(x):
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) == 0:
            raise ValueError("Expected a non-empty image list")
        return x[0]
    else:
        return x


def to_tuple(sz, dim, name):
    if isinstance(sz, numbers.Number):
        return (sz,) * dim
    if isinstance(sz, tuple):
        if len(sz) == 1:
            return sz * dim
        elif len(sz) == dim:
            return sz
    raise ValueError(f"Expected a number of {dim}-tuple for {name}")


def get_image_size(img):
    if isinstance(img, PIL.Image.Image):
        return (img.width, img.height)
    if isinstance(img, torch.Tensor):
        if img.ndim < 3:
            raise ValueError("Unsupported torch tensor (should have 3 dimensions or more)")
        return (img.shape[-1], img.shape[-2])
    if isinstance(img, np.ndarray):
        if img.ndim != 3:
            raise ValueError("Unsupported numpy array (should have 3 dimensions)")
        return (int(img.shape[1]), int(img.shape[0]))
    raise ValueError("Unsupported image type")


def resize(img, size):

    resizer = albumentations.Resize(height=size[0], width=size[0])
    if isinstance(img, np.ndarray):
#        rsz_img = F.resize(PIL.Image.fromarray(img), size)
#        np_img = np.array(rsz_img)
#        return np_img
        rsz_img = resizer(image=img)
        return rsz_img['image']
        

    return resizer(image=img)


def crop(img, top, left, height, width):
    """Torchvision crop + numpy ndarray support
    """
    if isinstance(img, np.ndarray):
        return PIL.Image.fromarray(img[top:top+height, left:left+width])
    return F.crop(img, top, left, height, width)


def rotate(img, degree):
    """Torchvision rotate + numpy ndarray support
    """
    if isinstance(img, np.ndarray):
        return F.rotate(PIL.Image.fromarray(img), degree)
    return F.rotate(img, degree)

def distort(img, k_1):
    h, w, c = img.shape
    deformer = BarrelDeformer(w, h, k_1)
    if isinstance(img, np.ndarray):
        pil_img = PIL.Image.fromarray(img)
#        pil_img.putalpha(255)

#        deform_img = ImageOps.deform(pil_img, deformer)
        return ImageOps.deform(pil_img, deformer)
#    img.putalpha(255)
    return ImageOps.deform(img, deformer)


def albu_distort(img, k_1):
#    h, w, c = img.shape
#    k_1 = 0.01 * (48/min(w,h)) * k_1
#    k_1 = (0, k_1*0.2)
#    k_1 = (k_1*0.2, k_1*0.2)
    deformer = albumentations.OpticalDistortion(distort_limit=k_1, shift_limit=0)
    if isinstance(img, np.ndarray):
#        pil_img = PIL.Image.fromarray(img)
#        pil_img.save(f'albudistort{k_1}_before.png')
        deform_img = deformer(image=img)
        pil_img = PIL.Image.fromarray(deform_img['image'])
#        pil_img.save(f'albudistort{k_1}_after.png')
#        breakpoint()
        return pil_img  
    return deformer(image=img)


def albu_shift(img, nvx, nvy):
    nvx = 0.01 * nvx
    nvy = 0.01 * nvy
    translater = albumentations.Affine(translate_percent={"x" : nvx, "y": nvy}, p=1)
    # translater = albumentations.Affine(translate_px={"x" : nvx, "y": nvy}, p=1)
    if isinstance(img, np.ndarray):
        translate_img = translater(image=img)
        pil_img = PIL.Image.fromarray(translate_img['image'])
        return pil_img
    else:
        translate_img = translater(image=img)
        pil_img = PIL.Image.fromarray(translate_img['image'])
        return translater(image=img)['image']

def albu_rotate(img, nvx, nvy):
    rotator = albumentations.Affine(rotate={"x" : nvx, "y": nvy})
    if isinstance(img, np.ndarray):
        rotate_img = rotator(image=img)
        pil_img = PIL.Image.fromarray(rotate_img['image'])
        return pil_img
    return rotator(image=img)['image']



def rot90(img):
    if isinstance(img, PIL.Image.Image):
        return img.transpose(PIL.Image.ROTATE_90)
    return torch.rot90(img, dims=(-2, -1))


def random_uniform(minval, maxval):
    return float(torch.empty(1).uniform_(minval, maxval))


def random_uniform_none(bounds):
    if bounds is None:
        return None
    return random_uniform(bounds[0], bounds[1])


def param_to_tuple(param, name, center=1.0, bounds=(0.0, float("inf"))):
    if isinstance(param, (list, tuple)):
        if len(param) != 2:
            raise ValueError(f"{name} must have two bounds")
        return (max(bounds[0], param[0]), min(bounds[1], param[1]))
    if not isinstance(param, numbers.Number):
        raise ValueError("f{name} must be a number or a pair")
    if param == 0:
        return None
    minval = max(center - param, bounds[0])
    maxval = min(center + param, bounds[1])
    return (minval, maxval)


def get_resize_params(scales):
    pixels = int(functools.reduce(np.lcm, [sc for sc in scales]))
    size_ratios = [pixels // sc for sc in scales]
    return size_ratios

def get_crop_params(x, scales):
    if not isinstance(x, (list, tuple)):
        # Just the image size with no scaling needed
        return get_image_size(x), [(1, 1)]
    assert len(x) == len(scales)
    sizes = [get_image_size(img) for img in x]
    # Find a size in which all images fit
    scaled_widths = [sc[0]*sz[0] for sc, sz in zip(scales, sizes)]
    scaled_heights = [sc[1]*sz[1] for sc, sz in zip(scales, sizes)]
    min_width = min(scaled_widths)
    min_height = min(scaled_heights)
    # Check that the scales are close enough to the actual sizes (5%)
    if max(scaled_widths) > min_width * 1.05:
        raise ValueError(
            f"Scaled widths range from {min_width} to {max(scaled_widths)}. "
            f"This does not seem compatible")
    if max(scaled_heights) > min_height * 1.05:
        raise ValueError(
            f"Scaled heights range from {min_height} to {max(scaled_heights)}. "
            f"This does not seem compatible")
    # Now find a size so that pixel-accurate cropping is possible for all images
    pixels_x = int(functools.reduce(np.lcm, [sc[0] for sc in scales]))
    pixels_y = int(functools.reduce(np.lcm, [sc[1] for sc in scales]))
    common_size = (min_width // pixels_x, min_height // pixels_y)
    size_ratios = [(pixels_x // sc[0], pixels_y // sc[1]) for sc in scales]
    return common_size, size_ratios


def check_size_valid(size, scales, name):
    width, height = size
    for ws, hs in scales:
        if width % ws != 0:
            raise ValueError(f"Scale {ws} is incompatible with {name} {width}")
        if height % hs != 0:
            raise ValueError(f"Scale {hs} is incompatible with {name} {height}")


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    def __call__(self, x):
        return apply_all(x, F.to_tensor)


class ToPILImage:
    def __call__(self, x):
        return apply_all(x, F.to_pil_image)


class AlbuRandomCrop_shift(nn.Module):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 shift_value: int = 1,
                 margin: float = 0.0):
        super(AlbuRandomCrop_shift, self).__init__()
        self.size = to_tuple(size, 2, "AlbuRandomCrop_shift.size")
        self.scales = [to_tuple(s, 2, "AlbuRandomCrop_shift.scale") for s in scales]
        self.shift_value = shift_value
        self.margin = margin
        check_size_valid(self.size, self.scales, "AlbuRandomCrop_shift.size")
        # TODO: other torchvision.transforms.RandomCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        margin_w = int(tw * self.margin)
        margin_h = int(th * self.margin)
        i = torch.randint(-margin_h, h - th + 1 + margin_h, size=(1, )).item()
        j = torch.randint(-margin_w, w - tw + 1 + margin_w, size=(1, )).item()
        i = np.clip(i, 0,  h - th)
        j = np.clip(j, 0,  w - tw)
        sv = self.shift_value
        shift_values = []
        shift_values.append((0, 0))
        mh = torch.randint(-sv, sv+1, size=(1, )).item()
        mw = torch.randint(-sv, sv+1, size=(1, )).item()
        shift_values.append((mw, mh))

        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh), (tw_tmp, th_tmp) in zip(x, size_ratios, shift_values):
            if tw_tmp != 0 or th_tmp != 0:
                img = albu_shift(img, tw_tmp * (tw)/(w), th_tmp * (th)/(h))
                # img = albu_shift(img, tw_tmp, th_tmp)
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        
        return ret

class AlbuRandomCrop_distort(nn.Module):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 distort_value: int = 1,
                 margin: float = 0.0, 
                 index_patch : bool = False):
        super(AlbuRandomCrop_distort, self).__init__()
        self.size = to_tuple(size, 2, "RandomCrop.size")
        self.scales = [to_tuple(s, 2, "RandomCrop.scale") for s in scales]
        self.distort_value = distort_value
        self.margin = margin
        self.index_patch = index_patch
        check_size_valid(self.size, self.scales, "RandomCrop.size")
        # TODO: other torchvision.transforms.RandomCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        margin_w = int(tw * self.margin)
        margin_h = int(th * self.margin)
        i = torch.randint(-margin_h, h - th + 1 + margin_h, size=(1, )).item()
        j = torch.randint(-margin_w, w - tw + 1 + margin_w, size=(1, )).item()
        i = np.clip(i, 0,  h - th)
        j = np.clip(j, 0,  w - tw)
#        dvw = torch.randint(-self.distort_value, self.distort_value+1, size=(1, )).item()
#        dvh = torch.randint(-self.distort_value, self.distort_value+1, size=(1, )).item()
#        dv = torch.randint(0, self.distort_value+1, size=(1, )).item()
#        dv = torch.randint(0, int(self.distort_value*min(w,h)/48)+1, size=(1, )).item()
#        dv = self.distort_value * 0.01 * (48/min(w,h))

#        if self.index_patch :
#            pi = w // i
#            pj = h // j

        distort_values = []
        distort_values.append(0)
        distort_values.append(self.distort_value)
#        distort_values.append((0,0))
#        distort_values.append((0,dv))

        
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh), dv_tmp in zip(x, size_ratios, distort_values):
#        for img, (rw, rh), (dvl_tmp, dvh_tmp) in zip(x, size_ratios, distort_values):
            if dv_tmp != 0 :
#                pilimg = PIL.Image.fromarray(img)
                img = albu_distort(img, dv_tmp)
#                img = albu_distort(img, (dv_tmp, dv_tmp))
#               img = albu_distort(img, (dvl_tmp, dvh_tmp))
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        
        return ret

class RandomCrop_distort(nn.Module):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 distort_value: int = 1,
                 margin: float = 0.0):
        super(RandomCrop_distort, self).__init__()
        self.size = to_tuple(size, 2, "RandomCrop.size")
        self.scales = [to_tuple(s, 2, "RandomCrop.scale") for s in scales]
        self.distort_value = distort_value
        self.margin = margin
        check_size_valid(self.size, self.scales, "RandomCrop.size")
        # TODO: other torchvision.transforms.RandomCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        margin_w = int(tw * self.margin)
        margin_h = int(th * self.margin)
        i = torch.randint(-margin_h, h - th + 1 + margin_h, size=(1, )).item()
        j = torch.randint(-margin_w, w - tw + 1 + margin_w, size=(1, )).item()
        i = np.clip(i, 0,  h - th)
        j = np.clip(j, 0,  w - tw)
#        dvw = torch.randint(-self.distort_value, self.distort_value+1, size=(1, )).item()
#        dvh = torch.randint(-self.distort_value, self.distort_value+1, size=(1, )).item()
        dv = torch.randint(0, self.distort_value+1, size=(1, )).item()
#        dv = self.distort_value * 0.01 * (48/min(w,h))
        distort_values = []
        distort_values.append(0)
        distort_values.append(dv)

        
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh), dv_tmp in zip(x, size_ratios, distort_values):
            if dv_tmp != 0 :
                pilimg = PIL.Image.fromarray(img)
                img = distort(img, dv_tmp)
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        
        return ret


class RandomCrop_rotate(nn.Module):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 rotate_value: int = 1,
                 margin: float = 0.0):
        super(RandomCrop_rotate, self).__init__()
        self.size = to_tuple(size, 2, "RandomCrop.size")
        self.scales = [to_tuple(s, 2, "RandomCrop.scale") for s in scales]
        self.rotate_value = rotate_value
        self.margin = margin
        check_size_valid(self.size, self.scales, "RandomCrop.size")
        # TODO: other torchvision.transforms.RandomCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        margin_w = int(tw * self.margin)
        margin_h = int(th * self.margin)
        i = torch.randint(-margin_h, h - th + 1 + margin_h, size=(1, )).item()
        j = torch.randint(-margin_w, w - tw + 1 + margin_w, size=(1, )).item()
        i = np.clip(i, 0,  h - th)
        j = np.clip(j, 0,  w - tw)
        rv = torch.randint(-self.rotate_value, self.rotate_value+1, size=(1, )).item()
        rotate_values = []
        rotate_values.append(0)
        rotate_values.append(rv)
        
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh), rv_tmp in zip(x, size_ratios, rotate_values):
            if rv_tmp != 0 :
                img = rotate(img, rv_tmp)
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        
        return ret


class RandomCrop_shift(nn.Module):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 shift_value: int = 1,
                 margin: float = 0.0):
        super(RandomCrop_shift, self).__init__()
        self.size = to_tuple(size, 2, "RandomCrop.size")
        self.scales = [to_tuple(s, 2, "RandomCrop.scale") for s in scales]
        self.shift_value = shift_value
        self.margin = margin
        check_size_valid(self.size, self.scales, "RandomCrop.size")
        # TODO: other torchvision.transforms.RandomCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        margin_w = int(tw * self.margin)
        margin_h = int(th * self.margin)
        i = torch.randint(-margin_h, h - th + 1 + margin_h, size=(1, )).item()
        j = torch.randint(-margin_w, w - tw + 1 + margin_w, size=(1, )).item()
        i = np.clip(i, 0,  h - th)
        j = np.clip(j, 0,  w - tw)
        sv = self.shift_value
        shift_values = []
        shift_values.append((0, 0))
        mh = torch.randint(-sv, sv+1, size=(1, )).item()
        mw = torch.randint(-sv, sv+1, size=(1, )).item()
        shift_values.append((mw, mh))
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh), (mw_tmp, mh_tmp) in zip(x, size_ratios, shift_values):
            i_tmp = i * rh + mh_tmp
            j_tmp = j * rw + mw_tmp
            i_tmp = np.clip(i_tmp, 0,  h - th)
            j_tmp = np.clip(j_tmp, 0,  w - tw)
            ret.append(crop(img, i_tmp, j_tmp, th * rh, tw * rw))
            # ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        
        return ret


class Resize(nn.Module):

    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 margin: float = 0.0):
        super(Resize, self).__init__()
        self.size = to_tuple(size, 2, "Resize.size")
        self.scales = [s for s in scales]
#        check_size_valid(self.size, self.scales, "Resize.size")

    def forward(self, x):
        scales = self.scales
        size_ratios = get_resize_params(scales)
        common_resize_size = (self.size[0] // size_ratios[0], self.size[1] // size_ratios[0])
        i , j = common_resize_size

        if not isinstance(x, (list, tuple)):
            return resize(x, self.size)
        imgs = []
        for img, r in zip(x, size_ratios):
            imgs.append(resize(img, (i*r, j*r)))
        
        return imgs


class RandomCrop(nn.Module):
    """Crop the given images at a common random location.

    The location is chosen so that all images are cropped at a pixel boundary,
    even if they have different resolutions.

    Args:
        size (int or tuple): Size to which the HR image will be cropped.
        scales (list): Scales of the images received.
        margin (float): Margin used to bias selection towards the borders of
            the image.
    """

    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 margin: float = 0.0):
        super(RandomCrop, self).__init__()
        self.size = to_tuple(size, 2, "RandomCrop.size")
        self.scales = [to_tuple(s, 2, "RandomCrop.scale") for s in scales]
        self.margin = margin
        check_size_valid(self.size, self.scales, "RandomCrop.size")
        # TODO: other torchvision.transforms.RandomCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        margin_w = int(tw * self.margin)
        margin_h = int(th * self.margin)
        i = torch.randint(-margin_h, h - th + 1 + margin_h, size=(1, )).item()
        j = torch.randint(-margin_w, w - tw + 1 + margin_w, size=(1, )).item()
        i = np.clip(i, 0,  h - th)
        j = np.clip(j, 0,  w - tw)
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh) in zip(x, size_ratios):
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        
        return ret


class AdjustToScale(nn.Module):
    """Crop the given images so that they match the scale exactly

    Args:
        scales (list): Scales of the images received.
    """

    def __init__(self,
                 scales: List[int]):
        super(AdjustToScale, self).__init__()
        self.scales = [to_tuple(s, 2, "AdjustToScale.scale") for s in scales]

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        tw, th = common_size
        if not isinstance(x, (list, tuple)):
            return crop(x, 0, 0, th, tw)
        ret = []
        for img, (rw, rh) in zip(x, size_ratios):
            ret.append(crop(img, 0, 0, th * rh, tw * rw))
        return ret


class CenterCrop(nn.Module):
    """Crop the center of the given images

    The location is chosen so that all images are cropped at a pixel boundary,
    even if they have different resolutions.

    Args:
        size (int or tuple): Size to which the HR image will be cropped.
        scales (list): Scales of the images received.
        allow_smaller (boolean, optional): Do not error on images smaller
            than the given size
    """

    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 allow_smaller: bool = False):
        super(CenterCrop, self).__init__()
        self.size = to_tuple(size, 2, "CenterCrop.size")
        self.allow_smaller = allow_smaller
        self.scales = [to_tuple(s, 2, "CenterCrop.scale") for s in scales]
        check_size_valid(self.size, self.scales, "CenterCrop.size")
        # TODO: other torchvision.transforms.CenterCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        # Check the size
        if th > h:
            if not self.allow_smaller:
                raise ValueError("Required height for CenterCrop is larger than the image")
            th = h
            i = 0
        else:
            i = (h - th) // 2
        if tw > w:
            if not self.allow_smaller:
                raise ValueError("Required height for CenterCrop is larger than the image")
            tw = w
            j = 0
        else:
            j = (w - tw) // 2
        # Apply
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh) in zip(x, size_ratios):
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        return ret
    

class Crop(nn.Module):
    """Crop the center of the given images

    The location is chosen so that all images are cropped at a pixel boundary,
    even if they have different resolutions.

    Args:
        size (int or tuple): Size to which the HR image will be cropped.
        scales (list): Scales of the images received.
        allow_smaller (boolean, optional): Do not error on images smaller
            than the given size
    """

    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 allow_smaller: bool = False):
        super(Crop, self).__init__()
        self.size = to_tuple(size, 2, "CenterCrop.size")
        self.allow_smaller = allow_smaller
        self.scales = [to_tuple(s, 2, "CenterCrop.scale") for s in scales]
        check_size_valid(self.size, self.scales, "CenterCrop.size")
        # TODO: other torchvision.transforms.CenterCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = (self.size[0] // crop_ratio[0], self.size[1] // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        # Check the size
        if th > h:
            if not self.allow_smaller:
                raise ValueError("Required height for CenterCrop is larger than the image")
            th = h
            i = 0
        else:
            i = (h - th) // 2 + 200
        if tw > w:
            if not self.allow_smaller:
                raise ValueError("Required height for CenterCrop is larger than the image")
            tw = w
            j = 0
        else:
            j = (w - tw) // 2 - 700
        # Apply
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh) in zip(x, size_ratios):
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        return ret


class EdgeRemove(nn.Module):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scales: List[int],
                 allow_smaller: bool = False):
        super(EdgeRemove, self).__init__()
        self.size = to_tuple(size, 2, "EdgeRemove.size")
        self.allow_smaller = allow_smaller
        self.scales = [to_tuple(s, 2, "EdgeRemove.scale") for s in scales]
        check_size_valid(self.size, self.scales, "EdgeRemove.size")
        # TODO: other torchvision.transforms.CenterCrop options

    def forward(self, x):
        scales = self.scales
        common_size, size_ratios = get_crop_params(x, scales)
        crop_ratio = size_ratios[0]
        common_crop_size = ((common_size[0]-self.size[0]) // crop_ratio[0], (common_size[1]-self.size[1]) // crop_ratio[1])
        w, h = common_size
        tw, th = common_crop_size
        # Check the size
        if th > h:
            if not self.allow_smaller:
                raise ValueError("Required height for CenterCrop is larger than the image")
            th = h
            i = 0
        else:
            i = (h - th) // 2
        if tw > w:
            if not self.allow_smaller:
                raise ValueError("Required height for CenterCrop is larger than the image")
            tw = w
            j = 0
        else:
            j = (w - tw) // 2
        # Apply
        if not isinstance(x, (list, tuple)):
            return crop(x, i, j, th, tw)
        ret = []
        for img, (rw, rh) in zip(x, size_ratios):
            ret.append(crop(img, i * rh, j * rw, th * rh, tw * rw))
        return ret



class ColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__()
        self.brightness = param_to_tuple(brightness, 'ColorJitter.brightness')
        self.contrast = param_to_tuple(contrast, 'ColorJitter.contrast')
        self.saturation = param_to_tuple(saturation, 'ColorJitter.saturation')
        self.hue = param_to_tuple(hue, 'ColorJitter.hue', center=0, bounds=[-0.5, 0.5])

    def get_params(self):
        brightness_factor = random_uniform_none(self.brightness)
        contrast_factor = random_uniform_none(self.contrast)
        saturation_factor = random_uniform_none(self.saturation)
        hue_factor = random_uniform_none(self.hue)
        return (brightness_factor, contrast_factor, saturation_factor, hue_factor)

    def apply_jitter(self, img, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        if brightness_factor is not None:
            img = F.adjust_brightness(img, brightness_factor)
        if contrast_factor is not None:
            img = F.adjust_contrast(img, contrast_factor)
        if saturation_factor is not None:
            img = F.adjust_saturation(img, saturation_factor)
        if hue_factor is not None:
            img = F.adjust_hue(img, hue_factor)
        return img

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params()
        return apply_all(x, lambda y: self.apply_jitter(y, brightness_factor, contrast_factor, saturation_factor, hue_factor))


### TODO SHKIM
class AlbuColorJitter(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(AlbuColorJitter, self).__init__()
        self.brightness = param_to_tuple(brightness, 'ColorJitter.brightness')
        self.contrast = param_to_tuple(contrast, 'ColorJitter.contrast')
        self.saturation = param_to_tuple(saturation, 'ColorJitter.saturation')
        self.hue = param_to_tuple(hue, 'ColorJitter.hue', center=0, bounds=[-0.5, 0.5])

    def get_params(self):
        brightness_factor = random_uniform_none(self.brightness)
        contrast_factor = random_uniform_none(self.contrast)
        saturation_factor = random_uniform_none(self.saturation)
        hue_factor = random_uniform_none(self.hue)
        return (brightness_factor, contrast_factor, saturation_factor, hue_factor)

    def apply_jitter(self, img, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        if brightness_factor is not None:
            img = F.adjust_brightness(img, brightness_factor)
        if contrast_factor is not None:
            img = F.adjust_contrast(img, contrast_factor)
        if saturation_factor is not None:
            img = F.adjust_saturation(img, saturation_factor)
        if hue_factor is not None:
            img = F.adjust_hue(img, hue_factor)
        return img

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params()
        return apply_all(x, lambda y: self.apply_jitter(y, brightness_factor, contrast_factor, saturation_factor, hue_factor))


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        if torch.rand(1) < self.p:
            x = apply_all(x, F.hflip)
        return x


class AlbuRandomHorizontalFlip(nn.Module):
    def __init__(self):
        super(AlbuRandomHorizontalFlip, self).__init__()

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        x = apply_all(x, albumentations.HorizontalFlip())
        return x


class RandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.p = p

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        if torch.rand(1) < self.p:
            x = apply_all(x, F.vflip)
        return x


class AlbuRandomVerticalFlip(nn.Module):
    def __init__(self):
        super(AlbuRandomVerticalFlip, self).__init__()

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        x = apply_all(x, albumentations.VerticalFlip())
        return x


class RandomFlipTurn(nn.Module):
    def __init__(self):
        super(RandomFlipTurn, self).__init__()

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        if torch.rand(1) < 0.5:
            x = apply_all(x, F.vflip)
        if torch.rand(1) < 0.5:
            x = apply_all(x, F.hflip)
        if torch.rand(1) < 0.5:
            x = apply_all(x, rot90)
        return x


class AlbuRandomFlipTurn(nn.Module):
    def __init__(self):
        super(AlbuRandomFlipTurn, self).__init__()

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        x = apply_all(x, albumentations.VerticalFlip())
        x = apply_all(x, albumentatinos.HorizontalFlip())
        x = apply_all(x, albumentations.RandomRatate90())
        return x

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=None, sigma=(0.1, 2.0), isotropic=False):
        super(GaussianBlur, self).__init__()
        self.kernel_size = None if kernel_size is None else to_tuple(kernel_size)
        self.sigma = param_to_tuple(sigma, 'GaussianBlur.sigma')
        self.isotropic = isotropic

    def forward(self, x):
        x = apply_all(x, remove_numpy)
        if self.isotropic:
            sigma_x = random_uniform(self.sigma[0], self.sigma[1])
            sigma_y = sigma_x
        else:
            sigma_x = random_uniform(self.sigma[0], self.sigma[1])
            sigma_y = random_uniform(self.sigma[0], self.sigma[1])
        sigma = (sigma_x, sigma_y)
        if self.kernel_size is not None:
            kernel_size = self.kernel_size
        else:
            k_x = max(2*int(math.ceil(3*sigma_x))+1, 3)
            k_y = max(2*int(math.ceil(3*sigma_y))+1, 3)
            kernel_size = (k_x, k_y)
        return apply_all(x, lambda y: F.gaussian_blur(y, kernel_size, sigma))
