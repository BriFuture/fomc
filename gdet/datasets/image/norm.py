import numpy as np
import cv2

def imnormalize(img: "np.ndarray", mean, std):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    img -= mean
    img *= stdinv
    # cv2.subtract(img, mean, img)  # inplace
    # cv2.multiply(img, stdinv, img)  # inplace
    return img