import torch
import torch.nn.functional as F
from ..bbox_dis import bbox_dis_points
from ..bbox_dis import get_inter_with_border_mul
import time
import numpy as np

#rcnn时候的nms
def lane_nms_rcnn(bboxes, scores, score_thr, nms_cfg, max_per_img, img_shape):
    """Performs non-maximum suppression in a batched fashion.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, 2).  lane和背景的分数
    Returns:
        tuple: kept dets and indice.
    """
#     import pdb
#     pdb.set_trace()
    nms_cfg_ = nms_cfg.copy()
    #添加offset使不同idx的相距很远，不能被nms去掉
    nms_type = nms_cfg_.pop('type', 'nms')
    #nms_op = eval(nms_type)
    #暂时只考虑lane，这一类，简化这部分
    scores = scores[:, 0]
    valid_mask = scores > score_thr
    boxes_for_nms = bboxes[valid_mask]
    scores_for_nms = scores[valid_mask]
    labels = scores_for_nms.new_zeros(scores_for_nms.shape[0])
    if boxes_for_nms.numel() == 0:
        bboxes = boxes_for_nms.new_zeros((0, 5))
        labels = boxes_for_nms.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels
    nms_op = lane_nms_gpu
    
    keep  = nms_op(boxes_for_nms, scores_for_nms, pad_shape=img_shape[:2], **nms_cfg_)
    boxes = boxes_for_nms[keep]
    scores = scores_for_nms[keep]
    
    return torch.cat((boxes, scores[:, None]), dim=-1), labels[keep]


def lane_batched_nms(boxes, scores, idxs, nms_cfg, img_shape, class_agnostic=False):
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
    nms_op = lane_nms_fast
    split_thr = nms_cfg_.pop('split_thr', 10000)
    #大于1万分层nms
    if len(boxes_for_nms) < split_thr:
        keep, roiToalign_total = nms_op(boxes_for_nms, scores, pad_shape=img_shape, **nms_cfg_)
        boxes = boxes[keep]
        scores = scores[keep]
        roiToalign = roiToalign_total[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        roiToalign_list=[]
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep, roiToalign = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            roiToalign_list.append(roiToalign)
        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        roiToalign = torch.cat(roiToalign_list)[keep]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep, roiToalign


def lane_nms_gpu_old(dets, scores, pad_shape, iou_threshold=0.05):
    """

    :param dets: (torch.Tensor): boxes in shape (N, 4).
    :param scores: (torch.Tensor): scores in shape (N, ).
    :param thresh:
    :return:
    """
    #print('nms_before_len', len(dets))
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.size()[0] > 0:
        i = order[0]
        keep.append(i.item())
        dis = bbox_dis(dets[i:i+1], dets[order[1:].cpu().numpy()], pad_shape).squeeze(0)
        inds = torch.nonzero(torch.gt(dis, iou_threshold)).squeeze(1)
        order = order[inds + 1]
    #print('nms_before_after', len(keep))
    return keep


#dis越大越好的nms
def lane_nms_gpu_new(dets, scores, pad_shape, iou_threshold=0.05):
    """

    :param dets: (torch.Tensor): boxes in shape (N, 4).
    :param scores: (torch.Tensor): scores in shape (N, ).
    :param thresh:
    :return:
    """
    #print('nms_before_len', len(dets))
    order = torch.argsort(scores, descending=True)
    keep = []
    # 减少重复的计算量，先得到所有proposal的inter——points
    
    while order.size()[0] > 0:
        i = order[0]
        keep.append(i.item())
        dis = bbox_dis(dets[i:i+1], dets[order[1:].cpu().numpy()], pad_shape).squeeze(0)
        inds = torch.nonzero(torch.lt(dis, iou_threshold)).squeeze(1)
        order = order[inds + 1]
    #print('nms_before_after', len(keep))
    return keep


#dis越大越好的nms, 先得到交点减少之前不必要的运算，并且返回
def lane_nms_gpu(dets, scores, pad_shape, iou_threshold=0.05, H=None, W=None, rcnn=False):
    """

    :param dets: (torch.Tensor): boxes in shape (N, 4).
    :param scores: (torch.Tensor): scores in shape (N, ).
    :param rcnn: 是否要接入rcnn，是的话输出转换后的anchor
    :param thresh:
    :return:
    """
    #print('nms_before_len', len(dets))
    order = torch.argsort(scores, descending=True)
    keep = []
    # 减少重复的计算量，先得到所有proposal的inter——points
    #得到dets与边界的交点 （N ,2)
    dets_inter_points,_,_,_ = get_inter_with_border_mul(dets, pad_shape[0], pad_shape[1])
    #两个个点（N ,2 ,2)
    dets_points = torch.stack((dets[:, :2], dets_inter_points), dim=1)
#     import pdb
#     pdb.set_trace()
    #CCTODO float() numpy item()是否可以去掉
    
    while order.size()[0] > 0:
        #count += 1
        i = order[0]
        keep.append(i)
        w_gap = torch.abs(dets[i:i+1 , 2].unsqueeze(1) - dets[order[1:] , 2].unsqueeze(0))
        w_gap1 = w_gap / dets[i:i+1 , 2].unsqueeze(1)
        w_dis = (w_gap1> 1/3).int()
        dis = (1 - bbox_dis_points(dets_points[i:i+1], dets_points[order[1:]], pad_shape) - w_dis).squeeze(0)
        inds = torch.nonzero(torch.lt(dis, iou_threshold)).squeeze(1)
        order = order[inds + 1]
    
#     elapsed = (time.time() - start)
#     print("nms Time used:{} bbox_dis used {}".format(elapsed, elapsed / count))
#     print('nms_before_after', len(keep))
#     if rcnn:
#         #得到输入到rroialignbbox
#         roi2align_x_y = (dets[:, :2] + dets_inter_points) / 2 
#         roi2align_h = F.pairwise_distance(dets[:, :2], dets_inter_points, keepdim = True) / 2
#         roi2align = torch.cat((roi2align_x_y, dets[:, 2:3], roi2align_h, dets[:, 3:4]), dim=1)
#         return torch.stack(keep), roi2align
    return torch.stack(keep)

def lane_nms_fast(dets, scores, thr = 0.0, pad_shape=None, H=None, W=None,  iou_threshold=0.05, rcnn=False):
    """
    dets: Tensor (N, (x, y, w, h))
    """
    order = torch.argsort(scores, descending=True)
    keep = []
    count=0
    start = time.time()
    time_1 = []
    time_2 = []
    time_3 = []
#     import pdb
#     pdb.set_trace()
    while order.size()[0] > 0:
        start_2 = time.time()
        count += 1
        i = order[0]
        keep.append(i)
        time_2.append(time.time() - start_2)
        #当时x距离大于100或者w差距大于16或角度差大于10度，时候保留下来
        #overlaps = (torch.abs((dets[order[1:], 0] - dets[i:i+1, 0])) > 100)
        start_3 = time.time()
#         overlaps = (torch.abs((dets[order[1:], 0] - dets[i:i+1, 0])) > 100)  | (torch.abs(dets[order[1:], 2] - dets[i:i+1, 2]) > 16) | (torch.abs(dets[order[1:], 3] - dets[i:i+1, 3]) > 10)
        overlaps = dets[order[1:], 0] - dets[i:i+1, 0]
        time_3.append(time.time() - start_3)
        start_1 = time.time()
        inds = torch.nonzero(overlaps).squeeze(1)
        order = order[inds + 1]
        time_1.append(time.time() - start_1)
    elapsed = (time.time() - start)
    print("nms Time used:{} overlaps used {}".format(elapsed, elapsed / count))
    print('time_1', np.array(time_1).mean())
    print('time_2', np.array(time_2).mean())
    print('time_3', np.array(time_3).mean())
    print('nms_after', len(keep))
    #print(keep)
    return torch.stack(keep), dets