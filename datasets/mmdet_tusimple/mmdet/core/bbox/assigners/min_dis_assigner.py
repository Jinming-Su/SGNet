import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result_min import AssignResultMin
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MinDisAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=False,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='Dis')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, pad_shape , gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()
        #CCTODO
        
        distances_s, distances_w = self.iou_calculator(gt_bboxes, bboxes, pad_shape)
        # if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
        #         and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
        #     if self.ignore_wrt_candidates:
        #         ignore_overlaps = self.iou_calculator(
        #             bboxes, gt_bboxes_ignore, mode='iof')
        #         ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
        #     else:
        #         ignore_overlaps = self.iou_calculator(
        #             gt_bboxes_ignore, bboxes, mode='iof')
        #         ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
        #     overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_distances(distances_s, distances_w,  gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_distances(self, distances_s, distances_w, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            distances (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n, 2).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = distances_s.size(0), distances_s.size(1)
        
        # 1. assign -1 by default
        assigned_gt_inds = distances_s.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            min_distances_s = distances_s.new_ones((num_bboxes, ))
            min_distances_w = distances_s.new_ones((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = distances.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResultMin(
                num_gts,
                assigned_gt_inds,
                min_distances_s,
                labels=assigned_labels)
        # for each anchor, which gt 和他distance最小
        # for each anchor, which anchor和他distance最小


        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        min_distances_s, argmin_distances_s = distances_s.min(dim=0)
        min_distances_w, argmin_distances_w = distances_s.min(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_min_distances_s, gt_argmin_distances_s = distances_w.min(dim=1)
        gt_min_distances_w, gt_argmin_distances_w = distances_w.min(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(min_distances_s > self.neg_iou_thr)] = 0
        # elif isinstance(self.neg_iou_thr, tuple):
        #     assert len(self.neg_iou_thr) == 2
        #     assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
        #                      & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        # pos_inds = max_overlaps >= self.pos_iou_thr
        # assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
        
        # print(distances_w[argmin_distances_s]
#         pos_inds = (min_distances_s <= self.pos_iou_thr) & \
#                    (gt_min_distances_w[argmin_distances_s] == distances[argmin_distances_s, range(num_bboxes)])
                   #gt对应的anchor的最小距离                      #gt与与他s最小的anchor的距离
        # CCTODO check correvtness  阈值判断还有待再次确认
        #求出每个anchor与其面积最小的gt的w差值
        distance_to_gt = torch.gather(distances_w, dim=0, index = argmin_distances_s.unsqueeze(0))
        pos_inds = (min_distances_s <= self.pos_iou_thr) & (distance_to_gt.squeeze(0) < 1/3)
        
        #print(pos_inds)
        #CCTODO
        #assigned_gt_inds 每个acnhor对应的gt编号
        #print(argmin_distances_s.shape)
        assigned_gt_inds[pos_inds] = argmin_distances_s[pos_inds] + 1
        #CCTODO 暂时忽略
        # if self.match_low_quality:
        #     # Low-quality matching will overwirte the assigned_gt_inds assigned
        #     # in Step 3. Thus, the assigned gt might not be the best one for
        #     # prediction.
        #     # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
        #     # bbox 1 will be assigned as the best target for bbox A in step 3.
        #     # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
        #     # assigned_gt_inds will be overwritten to be bbox B.
        #     # This might be the reason that it is not used in ROI Heads.
        #     for i in range(num_gts):
        #         if gt_max_overlaps[i] >= self.min_pos_iou:
        #             if self.gt_max_assign_all:
        #                 max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
        #                 assigned_gt_inds[max_iou_inds] = i + 1
        #             else:
        #                 assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
        # CCTODO check gt_labels
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResultMin(
            num_gts, assigned_gt_inds, min_distances_s, labels=assigned_labels)
