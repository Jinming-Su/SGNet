import torch

from mmdet.utils import util_mixins


class AssignResultMin(util_mixins.NiceRepr):
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        min_distances (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> min_distances = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, min_distances, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), min_distances.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), min_distances.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, min_distances, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.min_distances = min_distances
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'min_distances': self.min_distances,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.min_distances is None:
            parts.append(f'min_distances={self.min_distances!r}')
        else:
            parts.append('min_distances.shape='
                         f'{tuple(self.min_distances.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.

        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assinged to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state

        Returns:
            :obj:`AssignResult`: Randomly generated assign results.

        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        from mmdet.core.bbox import demodata
        rng = demodata.ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            min_distances = torch.zeros(num_preds, dtype=torch.float32)
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = torch.zeros(num_preds, dtype=torch.int64)
            else:
                labels = None
        else:
            import numpy as np
            # Create an overlap for each predicted box
            min_distances = torch.from_numpy(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = torch.from_numpy(rng.rand(num_preds) < p_assigned)
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = torch.from_numpy(
                rng.rand(num_preds) < p_ignore) & is_assigned

            gt_inds = torch.zeros(num_preds, dtype=torch.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = torch.from_numpy(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]

            gt_inds = torch.from_numpy(
                rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            min_distances[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = torch.zeros(num_preds, dtype=torch.int64)
                else:
                    labels = torch.from_numpy(
                        # remind that we set FG labels to [0, num_class-1]
                        # since mmdet v2.0
                        # BG cat_id: num_class
                        rng.randint(0, num_classes, size=num_preds))
                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, min_distances, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.min_distances = torch.cat(
            [self.min_distances.new_zeros(len(gt_labels)), self.min_distances])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
