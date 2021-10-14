# Copyright (c) Jianqi Ma, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from maskrcnn_benchmark import _C


class _RROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):

        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        total_output = _C.rroi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1]
        )

        output, con_idx_x, con_idx_y = total_output
        ctx.save_for_backward(roi, con_idx_x, con_idx_y)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, con_idx_x, con_idx_y = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.rroi_align_backward(
            grad_output,
            rois,
            con_idx_x,
            con_idx_y,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w
        )
        return grad_input, None, None, None


rroi_align = _RROIAlign.apply


class RROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(RROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):

        # input roi form: [0, x, y, w, h, a], reverse (w, h) to (h, w)
        # rois_np = rois.data.cpu().numpy()
        # rois_np[:, 3:5] = rois_np[:, 4:2:-1]

#         k, x, y, w, h, a =  rois.unbind(dim=1)
#         rois_reverse = torch.cat([k[:, None], x[:, None], y[:, None], h[:, None], w[:, None], a[:, None]], dim=1)

        # input = input.cpu().cuda()

        return rroi_align(
            input, rois, self.output_size, self.spatial_scale
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import cv2
    import numpy as np
    from PIL import Image
    # from torch.autograd import Variable
    # from rroi_pooling.modules.rroi_pool import RRoIPool

    # import network
    imname = 'timg.jpeg'

    im = cv2.imread(imname)
    im = np.zeros((2000, 2000, 3), dtype=np.uint8)
    cv2.line(im, (1353, 500), (1000, 853), (255, 0, 255), 3)
    cv2.line(im, (1000, 853), (1000-353, 500), (255, 0, 255), 3)
    cv2.line(im, (1000-353, 500), (1000, 500-353), (255, 0, 255), 3)
    cv2.line(im, (1000, 500-353), (1353, 500), (255, 0, 255), 3)
    cv2.line(im, (1250, 500), (1000, 750), (255, 0, 255), 3)
    cv2.line(im, (1000, 750), (750, 500), (255, 0, 255), 3)
    cv2.line(im, (750, 500), (1000, 250), (255, 0, 255), 3)
    cv2.line(im, (1000, 250), (1250, 500), (255, 0, 255), 3)

    # cv2.imshow('win1', im.copy())

    ma = np.expand_dims(im, 0).transpose(0, 3, 1, 2)
    iminfo = np.array([im.shape[0], im.shape[1], 1])
    rois = np.array([[0, 1000, 500, 500, 200, 45], [0, 1000, 500, 500, 200, 0], [0, 1000, 500, 500, 500, 45]], dtype=np.float32)
    print('ma:', ma.shape, iminfo)

    ma = torch.tensor(ma).float().cuda()
    iminfo = torch.tensor(iminfo).cuda()#network.np_to_variable(iminfo, is_cuda=True)
    rois = torch.tensor(rois).cuda()#network.np_to_variable(rois, is_cuda=True)
    print('ma.requires_grad:', ma.requires_grad)
    ma.requires_grad = True
    rroi_pool = RROIAlign((100, 100), 1.0/1)

    pooled = rroi_pool(ma, rois)

    print('pooled:', pooled.size())

    crop = pooled.data.cpu().numpy()

    print(crop.shape, np.unique(crop))

    crop_trans = crop.transpose(0, 2, 3, 1)
    crop1 = Image.fromarray(crop_trans[0].astype(np.uint8))
    crop2 = Image.fromarray(crop_trans[1].astype(np.uint8))
    crop3 = Image.fromarray(crop_trans[2].astype(np.uint8))
    crop1.save('crop1.jpg', 'jpeg')
    crop2.save('crop2.jpg', 'jpeg')
    crop3.save('crop3.jpg', 'jpeg')

    print(crop.shape)

    mean = torch.mean(pooled)

    # mean.backward()

    grad = torch.autograd.grad(mean, ma)

    print(
    'grad:', type(grad), len(grad), np.unique(grad[0].data.cpu().numpy()), np.where(grad[0].data.cpu().numpy() > 0))

    #cv2.imshow('win2', np.array(crop.transpose(0, 2, 3, 1)[0], np.uint8))
    #cv2.imshow('win3', np.array(crop.transpose(0, 2, 3, 1)[1], np.uint8))
    #cv2.imshow('win4', np.array(crop.transpose(0, 2, 3, 1)[2], np.uint8))
    #cv2.waitKey(0)
