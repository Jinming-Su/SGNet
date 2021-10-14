import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from .builder import ANCHOR_GENERATORS

@ANCHOR_GENERATORS.register_module()
class LaneAnchorGenerator(object):
    """Standard anchor generator for 2D line-anchor-based detectors.

        Args:
            strides (list[int] | list[tuple[int, int]]): Strides of anchors
                in multiple feature levels in order (w, h).
            ratios (list[float]): The list of ratios between the height and width
                of anchors in a single level.
            scales (list[int] | None): Anchor scales for anchors in a single level.
                It cannot be set at the same time if `octave_base_scale` and
                `scales_per_octave` are set.
            angle (list[int] | None): Anchor angle for laneanchors in a single level.
                It cannot be set at the same time if `octave_base_scale` and
                `scales_per_octave` are set.
            base_sizes (list[int] | None): The basic sizes
                of anchors in multiple levels.
                If None is given, strides will be used as base_sizes.
                (If strides are non square, the shortest stride is taken.)
            scale_major (bool): Whether to multiply scales first when generating
                base anchors. If true, the anchors in the same row will have the
                same scales. By default it is True in V2.0
            octave_base_scale (int): The base scale of octave.
            scales_per_octave (int): Number of scales for each octave.
                `octave_base_scale` and `scales_per_octave` are usually used in
                retinanet and the `scales` should be None when they are set.
            centers (list[tuple[float, float]] | None): The centers of the anchor
                relative to the feature grid center in multiple feature levels.
                By default it is set to be None and not used. If a list of tuple of
                float is given, they will be used to shift the centers of anchors.
            center_offset (float): The offset of center in proportion to anchors'
                width and height. By default it is 0 in V2.0.
    """

    def __init__(self,
                 strides,
                 angles,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.angles = torch.Tensor(angles)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    angles=self.angles,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      angles,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """

        if center is None:
            x_center = self.center_offset * base_size
            y_center = self.center_offset * base_size
        else:
            x_center, y_center = center
        ws = base_size * scales
        centers = torch.tensor([x_center, y_center]).repeat(scales.shape[0], 1)
        ws = (base_size * scales).unsqueeze(1)
        cws = torch.cat([centers, ws], dim=-1)
        base_anchors = torch.cat([self._angle_enum(cw, angles) for cw in cws], dim=0)
        return base_anchors

    def _angle_enum(self, cw, angles):
        cw = cw.repeat(len(angles), 1).float()
        angles = angles.unsqueeze(1).float()
        return torch.cat([cw, angles], dim=1)

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

    def grid_anchors_on_line(self, featmap_sizes, line, device='cuda'):
        """Generate grid anchors in multiple feature levels based on line

                Args:
                    featmap_sizes (list[tuple]): List of feature map sizes in
                        multiple feature levels.
                    line(Tensor(2,)): 回归线的归一化水平坐标
                    device (str): Device where the anchors will be put on.
                Return:
                    list[torch.Tensor]: Anchors in multiple feature levels. \
                        The sizes of each tensor should be [N, 4], where \
                        N = width * height * num_base_anchors, width and height \
                        are the sizes of the corresponding feature level, \
                        num_base_anchors is the number of anchors for that level.
                """
 
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors_on_line(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                line = line.to(device),
                stride = self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors_on_line(self,
                                          base_anchors,
                                          featmap_size,
                                          line,
                                          stride=(16, 16),
                                          device='cuda'):

        feat_h, feat_w = featmap_size
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        #CCTODO check与提取cls_score间统一
        shift_y = ( (( torch.arange(0, feat_w, device=device) * (line[1] - line[0]) * feat_h / feat_w  + feat_h * line[0]).round()).clamp(0, feat_h - 1) * stride[0] ). type_as(shift_x)
        if torch.isnan(shift_y).sum() > 0:
            import pdb
            pdb.set_trace()
            print('*', shift_y, line, feat_h, stride[0])
        #print('shift_y', shift_y)
#         shift_y = (  torch.arange(0, feat_w, device=device) * (line[1] - line[0]) ) + (line[0]  * feat_h).repeat(feat_w)).type_as(shift_x).clamp(0, feat_h - 1) * stride[0]
        shift_w = torch.zeros_like(shift_x)
        shifts = torch.stack([shift_x, shift_y, shift_w, shift_w], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid
    
    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.
    
        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.
    
        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors



    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        feat_h, feat_w = featmap_size
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        shift_x = torch.arange(0, feat_w, device=device) * stride[1]
        shift_y = torch.arange(0, feat_h, device=device) * stride[0]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shift_ww = torch.zeros_like(shift_xx)
        shifts = torch.stack([shift_xx, shift_yy, shift_ww, shift_ww], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...

        return all_anchors