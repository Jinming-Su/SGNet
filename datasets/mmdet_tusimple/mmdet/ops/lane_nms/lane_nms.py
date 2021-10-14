import torch
from ..bbox_dis import bbox_dis

def lane_batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    #添加offset使不同idx的相距很远，不能被nms去掉
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_type = nms_cfg_.pop('type', 'nms')
    #nms_op = eval(nms_type)
    nms_op = lane_nms_gpu
    split_thr = nms_cfg_.pop('split_thr', 10000)
    #大于1万分层nms？，没太看懂
    if len(boxes_for_nms) < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def lane_nms_gpu(dets, scores, thresh=0.05):
    """

    :param dets: (torch.Tensor): boxes in shape (N, 4).
    :param scores: (torch.Tensor): scores in shape (N, ).
    :param thresh:
    :return:
    """
    order = torch.sargsort(scores, descending=True)
    keep = []
    while order.size()[0] > 0:
        i = order[0]
        keep.append(i)
        dis = bbox_dis(dets[i:i+1], dets[order[1:]]).unsqueeze(0)
        inds = torch.nonzero(torch.gt(dis, thresh)).unsqueeze(1)
        order = order[inds + 1]
    return keep