"""
  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  
      http://www.apache.org/licenses/LICENSE-2.0
  
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import cv2
import torchvision.transforms as T
import random
import torch
from text_conditioned_hflip import thflip  # load text_conditioned horizontal flip

color_keywords = ["white", "yellow", "blue", "red", "green", "black", "brown",
                "azure", "ivory", "teal", "silver", "purple", "gray", "orange",
                "maroon", "pink"]


def text_conditioned_color_jitter(image, target):
    """
    Text Conditioned Color Jittering (Rule Based)
    Input:
        image (np.array): H x W x C 
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    Return:
        aug_image:  (np.array): H x W x C
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    """
    caption = target['caption']
    skip_augm = False
    for keyword in color_keywords:
        if keyword in caption.split():
            skip_augm = True
            break  # skip augmentation
    if not skip_augm:
        aug_img = T.ColorJitter(brightness=.5, hue=.3)(image)
        return aug_img, target
    else:
        return image, target


def block_level_masking(image, target):
    """
    Apply Block Level Masking to input image
    Input:
        image (np.array): H x W x C 
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    Return:
        aug_image:  (np.array): H x W x C
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    """
    aug_img = T.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3))(image)
    return aug_img, target

def pixel_level_masking(image, target, scale_threshold=0.75):
    """
    Apply Pixel Level Masking to input image
    Input:
        image (np.array): H x W x C 
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    Return:
        aug_image:  (np.array): H x W x C
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    """
    pixel_mask = torch.bernoulli(torch.ones(image.shape) * 
                                 (1 - scale_threshold))
    aug_img = image * pixel_mask
    return aug_img, target

def gaussian_blur(image, target):
    """
    Apply Gaussian Blur to input image
    Input:
        image (np.array): H x W x C 
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    Return:
        aug_image:  (np.array): H x W x C
        target (dict): {'caption': str caption,
                        'boxes': list of boxes with format [x1, y1, x2, y2],
                        ...}
    """
    aug_img = T.GaussianBlur(kernel_size = (5,5), sigma = (0.01, 1))(image)
    return aug_img, target