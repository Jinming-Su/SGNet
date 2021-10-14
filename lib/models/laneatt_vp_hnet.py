import math

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34

from nms import nms
from lib.lane import Lane
from lib.focal_loss import FocalLoss

#from .resnet import resnet122 as resnet122_cifar
from .resnet import Resnet50
from .matching import match_proposals_with_targets
from time import *
from torch.nn import BatchNorm2d, BatchNorm1d
import torch.nn.functional as F
from .loss import OhemCELoss, OhemCELoss_weighted

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = True)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ASPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True, *args, **kwargs):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=18, padding=18)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan*5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan*4, out_chan, ks=1)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                    
class Decoder(nn.Module):
    def __init__(self, n_classes, low_chan=256, *args, **kwargs):
        super(Decoder, self).__init__()
        #self.conv_low = ConvBNReLU(low_chan, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(
                ConvBNReLU(256, 256, ks=3, padding=1),
                ConvBNReLU(256, 256, ks=3, padding=1),
                )
        self.conv_out_6classes = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, feat_aspp):
#         H, W = feat_low.size()[2:]
#         feat_low = self.conv_low(feat_low)
#         feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear',
#                 align_corners=True)
#         feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_aspp)
        logits = self.conv_out_6classes(feat_out)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                    
                    
class LaneATTVP_hnet(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 pretrained_backbone=False,
                 S=72,
                 img_w=640,
                 img_h=360,
                 w_interval=10.,
                 on_line=False,
                 weighted_loss = False,
                 anchors_freq_path=None,
                 topk_anchors=None,
                 anchor_feat_channels=64):
        super(LaneATTVP_hnet, self).__init__()
        # Some definitions
        self.img_w = img_w
        self.img_h = img_h
        self.feature_extractor, backbone_nb_channels, self.stride = get_backbone(backbone, pretrained_backbone)
        self.img_w = img_w
        self.n_strips = S - 1
        self.n_offsets = S
        self.fmap_h = img_h // self.stride
        fmap_w = img_w // self.stride
        self.fmap_w = fmap_w
        self.anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        self.h_conv1 = ConvBNReLU(backbone_nb_channels, 256, 1, 1, padding=0)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 6))
        
        
    def forward(self, x, rpn_proposals = None, vp=None, conf_threshold=None, nms_thres=0, nms_topk=3000):
        mini_batch, _, H, W = x.size()
        batch_features = self.feature_extractor(x)
        batch_features = self.h_conv1(batch_features )
        batch_features = nn.AdaptiveAvgPool2d((1,1))(batch_features)
        batch_features = batch_features.view(mini_batch, -1)
        out = self.head(batch_features)
        return 1, 1, 1 ,1, out

    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        h_paras = torch.cat((h_paras[:, :3], torch.zeros(h_paras.shape[0], 1, device = h_paras.device), h_paras[:,3:5], torch.zeros(h_paras.shape[0], 1, device = h_paras.device), h_paras[:, 5:6], torch.ones(h_paras.shape[0], 1, device = h_paras.device)), dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # apply confidence threshold
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list

    def loss(self, proposals_list, vp_logits=None, vp_preds=None, targets=None, vp_labels= None, lane_logits=None, lane_labels = None, h_paras = None,  cls_loss_weight=10, order=3):
        mse_loss = nn.MSELoss()
        h_paras = torch.cat((h_paras[:, :3], torch.zeros(h_paras.shape[0], 1, device = h_paras.device), h_paras[:,3:5], torch.zeros(h_paras.shape[0], 1, device = h_paras.device), h_paras[:, 5:6], torch.ones(h_paras.shape[0], 1, device = h_paras.device)), dim=1)
        h_paras = h_paras.reshape((-1, 3, 3))
        hnet_loss = 0
        valid_imgs = 0
        total_positives = 0
        for target, h_matrix in zip(targets, h_paras):
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                continue
            num_positives = target.shape[0]
            total_positives +=  num_positives
            #将proposals转换为self.proposals_to_pred(positives)普通的x,y形式
            targets_xy = self.proposals_to_pred_xy(target)
            ws = []
            hnet_loss_oneimg = 0
            for proposal_xy in targets_xy:
                if proposal_xy.shape[0] < 3:
                    continue
                proposal_xy = proposal_xy.T
                proposal_xy = torch.cat((proposal_xy, torch.ones(1, proposal_xy.shape[1], device = proposal_xy.device)), dim=0)
                transformed = h_matrix.mm(proposal_xy)
                if order == 3:
                    Y = torch.stack((torch.pow(transformed[1], 3), torch.pow(transformed[1], 2), transformed[1], torch.ones_like(transformed[1])), dim=1)
                elif order == 2:
                    Y = torch.stack((torch.pow(transformed[1], 2), transformed[1], torch.ones_like(transformed[1])), dim=1)
                elif order ==1 :
                    Y = torch.stack(( transformed[1], torch.ones_like(transformed[1])), dim=1)
                x_1 = transformed[0:1].T
                canshu = (torch.inverse((Y.T).mm(Y)).mm(Y.T)).mm(x_1)
                ws.append(canshu)
#                 import pdb
#                 pdb.set_trace()
                #x_1_8 = canshu[0] * transformed[1] + canshu[1]
                x_1_8 = Y.mm(canshu).squeeze(-1)
                p_1_8 = torch.stack((x_1_8, transformed[1], torch.ones_like(x_1_8)), dim=0)
                p_8 = torch.inverse(h_matrix).mm(p_1_8)
                hnet_loss_oneimg += mse_loss(p_8[0], proposal_xy[0])
            import pdb
            pdb.set_trace()
            hnet_loss += hnet_loss_oneimg / num_positives
            valid_imgs += 1
        # Batch mean
        hnet_loss /= valid_imgs
        loss = hnet_loss * 0.1
        return loss, {'hnet_loss': hnet_loss, 'batch_positives': total_positives}

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
    
    def generate_baseline_anchors(self):
        x_range = torch.arange(1, self.img_w // self.stride, 1) * self.stride
        y_range = torch.arange(1, self.img_h // self.stride, 1) * self.stride
        shift_xx, shift_yy = self._meshgrid(x_range, y_range) 
        shift_ww = torch.zeros_like(shift_xx)
        shifts = torch.stack([shift_xx, shift_yy, shift_ww], dim=-1)
        cws = torch.tensor([0.0, 0.0])
        angles = torch.arange(0.0, 180.0 + 10, 10).clamp(1, 179)
        cws = cws.repeat(len(angles), 1)
        angles = angles.unsqueeze(1).float()
        base_anchors = torch.cat([cws, angles], dim=1)
        #以一个点为矩形的anchors
        vp_base_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        base_anchors = vp_base_anchors.view(-1, 3)
        #转换为从图像边界出发
        x_y_angles = get_inter_with_border_mul(base_anchors, self.img_h, self.img_w)
        x_y_angles[:, :2]  = x_y_angles[:, :2] / torch.tensor([self.img_w, self.img_h], device = x_y_angles.device)
        return self.generate_anchor_parallel(x_y_angles, False), \
               self.generate_anchor_parallel(x_y_angles, True)
        
        
    def gen_vp_edge_anchors(self, vps):
        """

        :param vps: (batch_size, 2)
        :self.vp_base_anchors  : (n_proposal, 3)
        :return:
        """
        vps = torch.cat((vps, torch.zeros_like(vps[:, 0:1])), dim=-1)
        #(batch_size, n_proposal, (x, y, angle))
        vps_proposals = vps.unsqueeze(1) + self.vp_base_anchors.unsqueeze(0)
        x_y_angles = vps_proposals.reshape(-1, 3)
        #转换为从图像边界出发
        x_y_angles = get_inter_with_border_mul(x_y_angles, self.img_h, self.img_w)
        x_y_angles[:, :2]  = x_y_angles[:, :2] / torch.tensor([self.img_w, self.img_h], device = vps.device)
        return self.generate_anchor_parallel(x_y_angles, False), \
               self.generate_anchor_parallel(x_y_angles, True)

    def generate_anchor_parallel(self, x_y_angles, cut=False):

        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(x_y_angles.shape[0], 2 + 2 + 1 + self.fmap_h, device=x_y_angles.device)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(x_y_angles.shape[0], 2 + 2 + 1 + self.n_offsets, device=x_y_angles.device)
        angle = x_y_angles[:, 2] * math.pi / 180.  # degrees to radians
        start_x, start_y = x_y_angles[:, 0], x_y_angles[:, 1]
        anchor[:, 2] = 1 - start_y
        anchor[:, 3] = start_x
        anchor[:, 5:] = (start_x * self.img_w).unsqueeze(1) - (
                    1 - anchor_ys.unsqueeze(0) - 1 + start_y.unsqueeze(1)) / torch.tan(angle).unsqueeze(1) * self.img_h

        return anchor

    def gen_vp_base_anchors(self, angles, grid_pixels, step, on_line=False):
        x_range = torch.arange(-grid_pixels, grid_pixels + step, step)
        if on_line:
            y_range = torch.arange(-0, 0 + step, step)
        else:
            y_range = torch.arange(-grid_pixels, grid_pixels + step, step)
        shift_xx, shift_yy = self._meshgrid(x_range, y_range)
        shift_ww = torch.zeros_like(shift_xx)
        shifts = torch.stack([shift_xx, shift_yy, shift_ww], dim=-1)
        cws = torch.tensor([0.0, 0.0])
        cws = cws.repeat(len(angles), 1)
        angles = angles.unsqueeze(1).float()
        base_anchors = torch.cat([cws, angles], dim=1)
        #以一个点为矩形的anchors
        vp_base_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        vp_base_anchors = vp_base_anchors.view(-1, 3)
        return vp_base_anchors
    
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

    def proposals_to_pred_xy(self, proposals):
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
                
            points = torch.stack((lane_xs * self.img_w, lane_ys * self.img_h), dim=1).float()
            #points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
#             import pdb
#             pdb.set_trace()
#             lane = Lane(points=points.cpu().numpy(),
#                         metadata={
#                             'start_x': lane[3],
#                             'start_y': lane[2],
#                             'conf': lane[1]
#                         })
            lanes.append(points)
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
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
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
    k = torch.tan(bboxes[:, 2] * np.pi / 180)
    mask1 = ((bboxes[:, 2] >= 90) & (k >= k1)).reshape((-1, 1))
    mask3 = ((bboxes[:, 2] <  90) & (k <=k2)).reshape((-1, 1))
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

    return torch.cat((inter_p, bboxes[:, 2:3]), 1)
