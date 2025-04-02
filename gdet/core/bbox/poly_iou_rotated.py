from gdet.ops.box_iou_rotated import box_iou_rotated
from .rotate_transforms import poly2obb_np, poly2obb
import numpy as np
import torch

def poly_iou_rotated_np(poly1: "np.ndarray", poly2: "np.ndarray", angle_version='oc'):
    """poly1: T[n, 8] poly2: T[n, 8]
    """
    obb1 = []
    for p1 in poly1:
        obb = poly2obb_np(p1, version=angle_version)
        if obb is None:
            obb = (0, 0, 1, 1, 0)
        obb1.append(obb)
    obb1 = np.asarray(obb1)
    obb2 = []
    for p2 in poly2:
        obb = poly2obb_np(p2, version=angle_version)
        if obb is None:
            obb = poly2obb_np(p2, version=angle_version)
            obb = (0, 0, 1, 1, 0)
        obb2.append(obb)
    obb2 = np.asarray(obb2)

    ious = box_iou_rotated_np(obb1, obb2)
    return ious

def poly_iou_rotated(poly1: "torch.Tensor", poly2: "torch.Tensor", angle_version='oc'):
    obb1 = poly2obb(poly1, version=angle_version)
    obb2 = poly2obb(poly2, version=angle_version)
    pass
    ious = box_iou_rotated(obb1, obb2)
    return ious

def box_iou_rotated_np(obb1: "np.ndarray", obb2: "np.ndarray"):
    ious = box_iou_rotated(
        torch.from_numpy(obb1).float(),
        torch.from_numpy(obb2).float()
    )
    return ious.numpy()


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

polygon_inter_union_cuda_enable = False
def polygon_inter_union_cpu(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """
    
    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros(n, m)
    union = torch.zeros(n, m)
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4,2)).convex_hull
        for j in range(m):
            polygon2 = shapely.geometry.Polygon(boxes2[j, :].view(4,2)).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured')
    return inter, union


def polygon_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu"):
    """
        Compute iou of polygon boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
    """
    
    boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)
    if torch.cuda.is_available() and polygon_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0
    
    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0] # 1xn
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0] # 1xn
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0] # 1xm
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0] # 1xm
        for i in range(boxes1.shape[0]):
            cw = torch.max(b1_x2[i], b2_x2) - torch.min(b1_x1[i], b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2[i], b2_y2) - torch.min(b1_y1[i], b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1[i] - b1_x2[i]) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1[i] - b1_y2[i]) ** 2) / 4  # center distance squared
                if DIoU:
                    iou[i, :] -= rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    w2, h2 = b2_x2-b2_x1, b2_y2-b2_y1+eps
                    w1, h1 = b1_x2[i]-b1_x1[i], b1_y2[i]-b1_y1[i]+eps
                    v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou[i, :] + (1 + eps))
                    iou[i, :] -= (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou[i, :] -= (c_area - union[i, :]) / c_area  # GIoU
    return iou  # IoU
