import torch
import torch.nn.functional as F

def masked_l1(x, y, mask):
    return (F.l1_loss(x, y, reduction='none') * mask).sum() / mask.sum()


def pad_by_divisor(t, div=32):
    C, H, W = t.shape
    Hp = div * (1 + H // div)
    Wp = div * (1 + W // div)
    
    new_t = torch.zeros(C, Hp, Wp)
    new_t[:, :H, :W] = t
    return new_t
    

def bbox_area(x0, y0, x1, y1):
    return (x1 - x0) * (y1 - y0)


def bbox_iou(a, b):
    a_x0, a_y0, a_x1, a_y1 = [e.item() for e in a]
    b_x0, b_y0, b_x1, b_y1 = [e.item() for e in b]

    x_min = max(a_x0, b_x0)
    y_min = max(a_y0, b_y0)
    x_max = min(a_x1, b_x1)
    y_max = min(a_y1, b_y1)

    I = bbox_area(x_min, y_min, x_max, y_max)
    A = bbox_area(a_x0, a_y0, a_x1, a_y1)
    B = bbox_area(b_x0, b_y0, b_x1, b_y1)

    return I / (A + B + 1e-6)

