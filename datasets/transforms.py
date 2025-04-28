import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CropBboxWithArea(object):
    def __init__(self, pixel=0):
        """Crop part of the image inside the bounding box
        The bbox should be in [xyxy] format
        The Image should be numpy array of shape [width, height, channel]

        Args:
            pixel (int, optional): The width of the zone surrounding the bounding box. Defaults to 0.
        """
        self.pixel = pixel
        
    def __call__(self, x, bbox):
        channel, height, width = x.shape
        x1, y1, x2, y2 = bbox
        new_x1 = max(x1 - self.pixel, 0)
        new_y1 = max(y1 - self.pixel, 0)
        new_x2 = min(x2 + self.pixel, width)
        new_y2 = min(y2 + self.pixel, height)
        x = x[:, new_y1:new_y2, new_x1:new_x2]
        return x
    


class PadSquared:
    def __init__(self, pad_value=0):
        """Pad the image into a squared image while keeping the aspect ratio.

        Args:
            pad_value (int, optional): The value to use for padding. Defaults to 0.
        """
        self.pad_value = pad_value

    def __call__(self, image):
        channel, height, width = image.shape
        new_shape = max(height, width)
        
        # Initialize padded array with the specified value
        padded_image = np.full((new_shape, new_shape, channel), self.pad_value, dtype=x.dtype)
        
        # Compute centering offsets
        x_center = (new_shape - height) // 2
        y_center = (new_shape - width) // 2
        
        # Place the original image in the center
        padded_image[:, y_center:y_center+height, x_center:x_center+width] = image
        
        return padded_image