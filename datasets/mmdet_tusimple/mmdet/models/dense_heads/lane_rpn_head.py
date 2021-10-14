import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmdet.core.bbox import lane_batched_nms
from mmdet.core import multi_apply
from ..builder import HEADS
from .lane_anchor_head import LaneAnchorHead
from .rpn_test_mixin import LaneRPNTestMixin
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import time

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap)
from mmdet.core import nms
from mmdet.core.bbox.bbox_dis.bbox_dis import get_roi2align
from ..builder import HEADS, build_loss
from maskrcnn_benchmark.structures.rboxlist_ops import box_nms
torch.set_printoptions(profile="full")
def line_smooth_l1_loss(target, pred,  reduce = 'mean', sigma=1):

    y1_loss = nn.SmoothL1Loss()(pred[..., 0], target[..., 0])
    y2_loss = nn.SmoothL1Loss()(pred[..., 1], target[..., 1])
    return y1_loss + y2_loss

@HEADS.register_module()
class LaneRPNHead(LaneRPNTestMixin, LaneAnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, **kwargs):
        super(LaneRPNHead, self).__init__(
            1, in_channels, background_label=0, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.line_conv_regress = nn.Sequential(
            nn.Conv2d(self.in_channels, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1))
        self.line_fc_regress = nn.Sequential(
            nn.Linear(256, 256),
            nn.Linear(256, 2))
        #self.line_conv_regress = nn.Conv2d(self.in_channels, 256, 3, padding=1)
        #self.line_fc_regress = nn.Linear(256, 2)
        self.line_act = nn.Sigmoid()

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
#         normal_init(self.line_conv_regress, std=0.01)
#         normal_init(self.line_fc_regress, std=0.01)

    def forward_single(self, x, lines):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        # whether inplace  gai before inplace
        x = F.relu(x, inplace=False)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        H, W = x.shape[-2:]
        x_inds = torch.arange(W, device = x.device)
        rpn_bbox_pred_list = []
        rpn_cls_score_list = []
        #对不同的线进行不同的处理
        for i in range(x.size(0)):
            
            y_inds = (x_inds * (lines[i, 1] - lines[i, 0]) * H / W + H * lines[i, 0]).round().long().clamp(0, H - 1)
#             import pdb
#             pdb.set_trace()
            #print('y_inds', y_inds, rpn_cls_score[i].shape)
            if ((y_inds<0) | (y_inds >=H)).sum() >0:
                print('h,w ', H, W, lines[i], y_inds, x_inds)
            assert ((y_inds<0) | (y_inds >=H)).sum() <=0
            #print('y_inds', y_inds, rpn_cls_score[i].shape)
            #提取出处于线上的特征向量     squeeze成为 (C,1,W)
            rpn_cls_score_list.append(rpn_cls_score[i][..., y_inds, x_inds].unsqueeze(-2))
            rpn_bbox_pred_list.append(rpn_bbox_pred[i][..., y_inds, x_inds].unsqueeze(-2))
        return torch.stack(rpn_cls_score_list), torch.stack(rpn_bbox_pred_list)
    

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            水平线结果（Tensor， (N, 2)与通常的分类、回归结果
            tuple: A tuple of classification scores and bbox prediction.

                -  cls_scores(list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        lines_list=[]
        lines_list = []
        pred_lines = self.forward_line(feats)
        #print('pred_lines', pred_lines)
        for _ in range(len(feats)):
            lines_list += [pred_lines]
        return (pred_lines, ) + multi_apply(self.forward_single, feats, lines_list)

    def forward_line(self, x):
        """Forward features from the upstream network.

                Args:
                    feats (tuple[Tensor]): Features from the upstream network, each is
                        a 4D-tensor.

                Returns:
                    tuple: A tuple of classification scores and bbox prediction.

                        - cls_scores (Tensor) : (N, 2)每张图片的水平线回归结果
                """
        line = self.line_conv_regress(x[-1])
        line = nn.AdaptiveAvgPool2d((1, 1))(line)
        line = line.view(line.size(0), -1)
        line = self.line_fc_regress(line)
        line = self.line_act(line)
        if torch.isnan(line).sum() > 0:
            import pdb
            pdb.set_trace()
        return line

    def forward_train(self,
                      x,
                      img_metas,
                      gt_lines,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, ).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        lines, cls_scores, bbox_preds = self(x)
        if gt_labels is None:
            loss_inputs = (lines, cls_scores, bbox_preds) + (gt_lines, gt_bboxes, img_metas)
        else:
            loss_inputs = (lines, cls_scores, bbox_preds) + (gt_lines, gt_bboxes, gt_labels, img_metas)
        #将outs中的cls_scores, bbox_preds根据line提取出来
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, featmap_sizes = featmap_sizes)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(lines, cls_scores, bbox_preds, img_metas, featmap_sizes=featmap_sizes, cfg=proposal_cfg)
            return losses, proposal_list
  

    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          gt_bboxes,
    #          img_metas,
    #          line,
    #          gt_bboxes_ignore=None):
    #     """Compute losses of the head.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W)
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W)
    #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
    #             shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         line (Tensor): line cordinate with shape（N，2）
    #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
    #             boxes can be ignored when computing the loss.
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     losses = super(LaneRPNHead, self).loss(
    #         cls_scores,
    #         bbox_preds,
    #         gt_bboxes,
    #         None,
    #         img_metas,
    #         line,
    #         gt_bboxes_ignore=gt_bboxes_ignore)
    #     return dict(
    #         loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'lines'))
    def loss(self,
             lines,
             cls_scores,
             bbox_preds,
             gt_lines,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None,
             featmap_sizes=None):
        """Compute losses of the head.

        Args:
            lines : tensor with shape (N, 2)
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, 1, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, 1, W)
            gt_annos 变成新的加上线的注释 list[tuple(line_tensor, bbox_tensor)]  line_tensor 是每张图片的水平线坐标[y1, y2],
             bbox_tensor是每张图片的所有车道线 （num_gts, 4) in (x, y, w, angle_x)
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            line (Tensor): line cordinate with shape（N，2）
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        line_loss = line_smooth_l1_loss(lines, torch.stack(gt_lines)) * 10
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, lines, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        #
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        #return dict(line_loss=line_loss)
        return dict(line_loss=line_loss, loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, 1, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, 1, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox


    def get_anchors(self, featmap_sizes, img_metas, lines, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

#         multi_level_anchors = self.anchor_generator.grid_anchors_on_line(
#             featmap_sizes, lines[0], device)
#         anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        # 和之前的区别是每张图片的anchor都不一样了
#         for i in range(num_imgs):
        anchor_list = []
        anchor_list = [self.anchor_generator.grid_anchors_on_line(
            featmap_sizes, line, device) for line in lines]


        valid_flag_list = []
        # CCTODO check flags and anchors
        stride = self.anchor_generator.strides
        #不同图片的anchor
        for img_id, anchor_img in enumerate(anchor_list):
            valid_flag_list_multilevel=[]
            #一张图片不同level的anchor
            for i, anchor_level in enumerate(anchor_img):
                h ,w = img_metas[img_id]['pad_shape'][:2]
                #print(h, w)
                valid_h = min(h, featmap_sizes[i][0] * stride[i][1])
                valid_w = w
                valid_y = anchor_level[:, 1] <= valid_h
                valid_x = anchor_level[:, 0] <= valid_w
                valid = valid_x & valid_y
                if valid.sum() <= 0:
                    print('anchor_level', anchor_level)
                #print(valid)
                valid_flag_list_multilevel.append(valid)
            valid_flag_list.append(valid_flag_list_multilevel)

        return anchor_list, valid_flag_list

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        # 将一张image中所有level的anchor连成一个tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        #分不同image进行
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        # 根据每层anchor的数量，不同images的anchor分到不同的level
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, img_meta['img_shape'][:2], gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        #CCTODO sampler理论上不用改
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        #print('sampling_result', sampling_result)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
#         import pdb
#         pdb.set_trace()
        #CCTO OK 检查的anchor基本是符合正例的
#         print('gt', img_meta)
#         print('gt_bbox', gt_bboxes)
        
#         print('sampling_result', sampling_result)
        #print('pos_anchor_bbox_and_gt', torch.stack((sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes), dim=2))
        
        #CCTODO 大概check一下正确性
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes, )
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            #print(bbox_targets.dtype, pos_bbox_targets.dtype)
            
            #print(pos_bbox_targets[:100].type_as(anchors))

            bbox_targets[pos_inds, :] = pos_bbox_targets.type_as(anchors)
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        # 根据inside——flages 返回到之前的anchor——set
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels,
                num_total_anchors,
                inside_flags,
                fill=self.background_label)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   lines,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   featmap_sizes=None,
                   cfg=None,
                   rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            lines : 
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            featmap_sizes : featmap_sizes
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        #print(lines.shape, cls_scores[0].shape, bbox_preds[0].shape, cls_scores[1].shape, bbox_preds[1].shape)
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        #featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        #print('line', lines[0], featmap_sizes)
        #print('anchors', mlvl_anchors)
        result_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = self.anchor_generator.grid_anchors_on_line(featmap_sizes, lines[img_id], device=device)
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            pad_shape = img_metas[img_id]['pad_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, lines[img_id], img_shape, pad_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list
    
    #在一张图片内进行
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           line,
                           img_shape,
                           pad_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, 1, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, 1, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (x, y, w, angle) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        #不同level的anchor
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))
        
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        # CCTODO 考虑数据增强中的翻转等的复原, 在不修改coder的情况下暂时取目前图片的W, H
        # 修改y为由线上的x推出来  
        H, W = img_shape[:2]
        pad_H, pad_W = img_shape[:2]
        #proposals = proposals_tmp.clone()
        #proposals[:, 1] = ( proposals_tmp[:, 0] * (line[1]-line[0]) *  pad_H / pad_W + pad_H * line[0] ).clamp(0, H-1)
        ids = torch.cat(level_ids)
        # TODO: remove the hard coded nms type
        #nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr, rcnn = cfg.rcnn)
        nms_type = cfg.pop('nms_type', 'nms')
        if nms_type=='nms':
            dets, _ = nms(torch.cat([proposals, scores[:, None]], dim=-1), 0.5)
        else:
            roi2align = get_roi2align(proposals, H, W)
            dets_roi2align, keep = box_nms(torch.cat((roi2align, scores[:,None]), dim=-1), cfg.nms_thr)
            dets = proposals[keep]
        #如果需要继续接入rcnn，返回连接了能够输入roialign的结果
        return dets[:cfg.nms_post]
