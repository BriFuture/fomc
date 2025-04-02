import torch
def safe_divide(x: "torch.Tensor", y: "torch.Tensor", rtol=1e-5, atol=1e-8):
    """Computes a safe divide which returns 0 if the denominator is zero.

    Reference:
    https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan
    Args:
      x: A float of numerator.
      y: A float of denominator.
      rtol: The relative tolerance parameter. See numpy.isclose for more info.
      atol: The absolute tolerance parameter. See numpy.isclose for more info.

    Returns:
      z: output x / y or 0.
    """
    zero = torch.Tensor([0.0]).to(device=y.device, dtype=y.dtype)
    is_zero = torch.isclose(y, zero, rtol=rtol, atol=atol)
    safe_y  = torch.where(is_zero, torch.ones_like(y), y)
    return torch.where(is_zero, torch.zeros_like(x), x / safe_y)

