import math

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34

from nms import nms
from lib.lane import Lane
from lib.focal_loss import FocalLoss

from .resnet import resnet122 as resnet122_cifar
from .matching import match_proposals_with_targets
from time import *
class LaneATT(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 pretrained_backbone=False,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64):
        super(LaneATT, self).__init__()
        # Some definitions
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(backbone, pretrained_backbone)
        self.img_w = img_w
        self.img_h = img_h
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = img_h // self.stride
        fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels

        self.n_proposal = 4
        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

#         # Generate anchors
#         # anchor与anchor在feature map上的坐标
#         self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)


#         # Filter masks if `anchors_freq_path` is provided
#         if anchors_freq_path is not None:
#             anchors_mask = torch.load(anchors_freq_path).cpu()
#             assert topk_anchors is not None
#             ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
#             self.anchors = self.anchors[ind]
#             self.anchors_cut = self.anchors_cut[ind]

#         # Pre compute indices for the anchor pooling
#         # cut的时候将cut的坐标都变成n * 1维，比分batch cut更好
#         self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
#             self.anchor_feat_channels, fmap_w, self.fmap_h)

        self.cut_zs, self.cut_ys = self.compute_anchor_cut_indices_proposals_ys_zs(self.anchor_feat_channels, self.fmap_w, self.fmap_h)
        # Setup and initialize layers
        self.conv1 = nn.Conv2d(backbone_nb_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.fmap_h, self.n_offsets + 1)
        #self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, len(self.anchors) - 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.fmap_h, 4 - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def forward(self, x, rpn_proposals = None, conf_threshold=None, nms_thres=0, nms_topk=3000):
        
        import pdb
        pdb.set_trace()
        batch_features = self.feature_extractor(x)
        batch_features = self.conv1(batch_features)
        batch_size = len(x)
        anchors = torch.zeros((batch_size, 4, 2 + 2 + 1 + self.n_offsets), device = x.device)
        anchors_cut = torch.zeros((batch_size, 4, 2 + 2 + 1 + self.fmap_h), device = x.device)
        #begin_time = time()
        for batch_idx in range(batch_size):
            for proposal_i in range(4):
                anchors[batch_idx][proposal_i] = self.generate_anchor((rpn_proposals[batch_idx, proposal_i, 0] / (self.img_w), rpn_proposals[batch_idx, proposal_i, 1] / (self.img_h)),  rpn_proposals[batch_idx, proposal_i, 3], False)
                anchors_cut[batch_idx][proposal_i] = self.generate_anchor((rpn_proposals[batch_idx, proposal_i, 0] / (self.img_w ), rpn_proposals[batch_idx, proposal_i, 1] / (self.img_h)),  rpn_proposals[batch_idx, proposal_i, 3], True)
        
        cut_xs, invalid_mask = self.compute_anchor_cut_indices_proposals_xs(anchors, anchors_cut, 
            self.anchor_feat_channels, self.fmap_w, self.fmap_h)
        
        #end_time = time()
        #print('cut耗时', end_time - begin_time)
        
        #begin_time = time()
        # (batch_size, n_proposals, n_fmaps, self.fmap_h, 1())
        batch_anchor_features = self.cut_anchor_features_proposals(batch_features, cut_xs, invalid_mask, self.cut_zs, self.cut_ys)

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)

        # Add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(x.shape[0], self.n_proposal, -1)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], self.n_proposal , -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offsets to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.n_offsets), device=x.device)
        reg_proposals += anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg

        # Apply nms
        proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_threshold, anchors)
        #end_time = time()
        return proposals_list

    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold, anchors=None):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for batch_idx, (proposals, attention_matrix) in enumerate(zip(batch_proposals, batch_attention_matrix)):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            anchor = anchors[batch_idx]
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                #默认为none，不为none的话有问题
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], anchors[[]], attention_matrix[[]], None))
                    continue
                #keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                num_to_keep  = (anchor[:, 0]>=0).sum()
#                 if num_to_keep < 4:
#                     import pdb
#                     pdb.set_trace()
                keep = torch.arange(len(proposals), device=proposals.device)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, anchors[batch_idx][keep], attention_matrix, anchor_inds))

        return proposals_list


    def loss(self, proposals_list, targets, cls_loss_weight=10):
        begin_time = time()
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for (proposals, anchors, _, _), target in zip(proposals_list, targets):
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
            if len(proposals) == 0:
                continue
            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
#                 import pdb
#                 pdb.set_trace()
                try:
                    positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = match_proposals_with_targets(
                        self, anchors, target)
                except:
                    import pdb
                    pdb.set_trace()
            
            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                                   dtype=torch.int)  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # Loss calc
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss =  reg_loss + 0 * cls_loss
        end_time = time()
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}


    def compute_anchor_cut_indices(self, n_fmaps, fmaps_w, fmaps_h):
        # definitions
        n_proposals = len(self.anchors_cut)

        # indexing
        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        unclamped_xs = unclamped_xs.reshape(n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def compute_anchor_cut_indices_proposals(self, anchors, anchors_cut, n_fmaps, fmaps_w, fmaps_h):
        n_proposals = 4
        anchors_cut = anchors_cut.reshape(-1, anchors_cut.shape[-1])
        
        unclamped_xs = torch.flip((anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        cut_xs = cut_xs.reshape(anchors.shape[0], -1, 1)
        unclamped_xs = unclamped_xs.reshape(anchors.shape[0], n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)
        
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask
    
    def compute_anchor_cut_indices_proposals_xs(self, anchors, anchors_cut, n_fmaps, fmaps_w, fmaps_h):
        n_proposals = 4
        anchors_cut = anchors_cut.reshape(-1, anchors_cut.shape[-1])
        
        unclamped_xs = torch.flip((anchors_cut[:, 5:] / self.stride).round().long(), dims=(1,))
        unclamped_xs = unclamped_xs.unsqueeze(2)
        unclamped_xs = torch.repeat_interleave(unclamped_xs, n_fmaps, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, fmaps_w - 1)
        cut_xs = cut_xs.reshape(anchors.shape[0], -1, 1)
        unclamped_xs = unclamped_xs.reshape(anchors.shape[0], n_proposals, n_fmaps, fmaps_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > fmaps_w)

        return cut_xs, invalid_mask  
    
    def compute_anchor_cut_indices_proposals_ys_zs(self,  n_fmaps, fmaps_w, fmaps_h):
        n_proposals = 4     
        cut_ys = torch.arange(0, fmaps_h)
        cut_ys = cut_ys.repeat(n_fmaps * n_proposals)[:, None].reshape(n_proposals, n_fmaps, fmaps_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(n_fmaps).repeat_interleave(fmaps_h).repeat(n_proposals)[:, None]

        return cut_zs, cut_ys
    
    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features
    
    def cut_anchor_features_proposals(self, features, cut_xs, invalid_mask, cut_zs, cut_ys):
        # definitions
        batch_size = features.shape[0]
        n_proposals = 4
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.fmap_h, 1), device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[cut_zs, cut_ys, cut_xs[batch_idx]].view(n_proposals, n_fmaps, self.fmap_h, 1)
            rois[invalid_mask[batch_idx]] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 lenght, S coordinates, score[0] = negative prob, score[1] = positive prob
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut
    
    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx
        
    
    
    def gen_vp_base_anchors(self, angles):

        
        x_range = torch.arange(-20., 25., 5.) 
        y_range = torch.arange(-20., 25., 5.)
        shift_xx, shift_yy = self._meshgrid(x_range, y_range)
        shift_ww = torch.zeros_like(shift_xx)
        shifts = torch.stack([shift_xx, shift_yy, shift_ww, shift_ww], dim=-1)
        
        cws = torch.tensor([0.0, 0.0, 0.0])
        cws = cws.repeat(len(angles), 1)
        angles = angles.unsqueeze(1).float()
        base_anchors = torch.cat([cws, angles], dim=1)
        
        #以一个点为矩形的anchors
        vp_base_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return base_anchors
    
    def _angle_enum(self, cw, angles):
        cw = cw.repeat(len(angles), 1).float()
        angles = angles.unsqueeze(1).float()
        return torch.cat([cw, angles], dim=1)
    
    #start是归一化的
    def generate_anchor(self, start, angle, cut=False):
        
        if cut:
            if angle < -1000:
                return -1e6 * torch.ones(2 + 2 + 1 + self.fmap_h)
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            if angle < -1000:
                return -1e6 * torch.ones(2 + 2 + 1 +self.n_offsets)
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = start_x * self.img_w - (1 - anchor_ys - 1 + start_y) / math.tan(angle) * self.img_h

        return anchor
    
    def generate_anchor_paralle(self, start, angle, cut=False):
        
        if cut:
            
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = start_x * self.img_w - (1 - anchor_ys - 1 + start_y) / math.tan(angle) * self.img_h

        return anchor
    
    
    def draw_anchors(self, img_w, img_h, k=None):
        base_ys = self.anchor_ys.numpy()
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        i = -1
        for anchor in self.anchors:
            i += 1
            if k is not None and i != k:
                continue
            anchor = anchor.numpy()
            xs = anchor[5:]
            ys = base_ys * img_h
            points = np.vstack((xs, ys)).T.round().astype(int)
            for p_curr, p_next in zip(points[:-1], points[1:]):
                img = cv2.line(img, tuple(p_curr), tuple(p_next), color=(0, 255, 0), thickness=5)

        return img

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
#         device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.anchor_cut_ys = device_self.anchor_cut_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
#         device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
#         device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self


def get_backbone(backbone, pretrained=False):
    if backbone == 'resnet122':
        backbone = resnet122_cifar()
        fmap_c = 64
        stride = 4
    elif backbone == 'resnet34':
        back = resnet34(pretrained=pretrained)
        back.load_state_dict(torch.load('resnet34-333f7ec4.pth'))
        backbone = torch.nn.Sequential(*list(back.children())[:-2])
        fmap_c = 512
        stride = 32
    elif backbone == 'resnet18':
        backbone = torch.nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])
        fmap_c = 512
        stride = 32
    else:
        raise NotImplementedError('Backbone not implemented: `{}`'.format(backbone))

    return backbone, fmap_c, stride


def get_inter_with_border_mul(bboxes, H, W):
    """
    并行得到N个anchor与边界的交点
    bboxes : torch.Tensor (N, 4) in (x, y, w, a)
    """
    #bboxes = bboxes.float()
    k1 =  (H-1-bboxes[:, 1]) / (-bboxes[:, 0] + 1e-6)
    k2 = (H-1-bboxes[:, 1]) / (W-1-bboxes[:, 0] + 1e-6)
    k = torch.tan(bboxes[:, 3] * np.pi / 180)
    mask1 = ((bboxes[:, 3] >= 90) & (k >= k1)).reshape((-1, 1))
    mask3 = ((bboxes[:, 3] <  90) & (k <=k2)).reshape((-1, 1))
    mask2 = (~(mask1 | mask3)).reshape((-1, 1))
    #print('mask', mask1, mask2, mask3)
    #左边交点的y
    p_l = torch.zeros_like(bboxes[:, :2])
    p_d = torch.zeros_like(p_l)
    p_r = torch.zeros_like(p_l)
    p_l[:, 1] = -k*bboxes[:, 0] + bboxes[:, 1]
    #下边交点的x
    p_d[:, 1].fill_(H-1)
    p_d[:, 0] = (H-1-bboxes[:, 1]) / (k + 1e-6) + bboxes[:, 0]
    #右边交点的y
    p_r[:, 0].fill_(W-1)
    p_r[:, 1] = k*(W-1-bboxes[:, 0]) + bboxes[:, 1]
    inter_p = mask1 * p_l + mask2 * p_d + mask3 * p_r
    return inter_p, mask1, mask2, mask3