import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYWABBoxCoder(BaseBBoxCoder):
    """Delta XYWHA BBox coder

    this coder encodes bbox (x, y, w, a) into delta (dx, dy, dw, da) and
    decodes delta (dx, dy, dw,  da) back to original bbox (x, y, w, a).

    Args:
        target_means (Sequence[float]): denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 with_y=1):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.with_y = with_y

    def encode(self, bboxes, gt_bboxes, with_y=1):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds, with_y=1)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip)

        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.), with_y=1):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()

    px, py, pw, pa = (proposals[:, i] for i in range(4))
    gx, gy, gw, ga = (gt[:, i] for i in range(4))

    dx = (gx - px) / pw
    dy = (gy - py) / pw * with_y
    #print(dx, dy)
    dw = torch.log(gw / pw)
    da = (ga - pa) / 180 * np.pi

    deltas = torch.stack([dx, dy, dw, da], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    da = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    # Compute rotated angle of each roi
    pa = rois[:, 3].unsqueeze(1).expand_as(da)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, pw, dy)  # gy = py + ph * dy
    # Compute angle
    ga = da * 180 / np.pi + pa
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)
    rbboxes = torch.stack([gx, gy, gw, ga], dim=-1).view_as(deltas)
    return rbboxes
