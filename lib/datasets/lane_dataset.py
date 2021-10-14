import logging

import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from scipy.interpolate import InterpolatedUnivariateSpline
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from lib.lane import Lane

from .culane import CULane
from .tusimple import TuSimple
from .llamas import LLAMAS
from .ziyan import Ziyan
from .nolabel_dataset import NoLabelDataset

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LaneDataset(Dataset):
    def __init__(self,
                 S=72,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=False,
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()
        if dataset == 'tusimple':
            self.dataset = TuSimple(**kwargs)
        elif dataset == 'culane':
            self.dataset = CULane(**kwargs)
        elif dataset == 'ziyan':
            self.dataset = Ziyan(**kwargs)
        elif dataset == 'llamas':
            self.dataset = LLAMAS(**kwargs)
        elif dataset == 'nolabel_dataset':
            self.dataset = NoLabelDataset(**kwargs)
        else:
            raise NotImplementedError()
        self.n_strips = S - 1
        self.n_offsets = S
        self.normalize = normalize
        self.img_h, self.img_w = img_size
        self.strip_size = self.img_h / self.n_strips
        self.logger = logging.getLogger(__name__)

        # y at each x offset
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.transform_annotations()
        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation
        else:
            augmentations = []

        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        '''
        augmentations:
        - name: Affine
          parameters:
            translate_px:
             x: !!python/tuple [-25, 25]
             y: !!python/tuple [-10, 10]
            rotate: !!python/tuple [-6, 6]
            scale: !!python/tuple [0.85, 1.15]
        - name: HorizontalFlip
          parameters:
            p: 0.5
        '''
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_lanes = self.dataset.max_lanes

    @property
    def annotations(self):
        return self.dataset.annotations

    def transform_annotations(self):
        self.logger.info("Transforming annotations to the model's target format...")
        #print(self.transform_annotation)
        #print(self.dataset.annotations)
        self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))
        self.logger.info('Done.')

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']
#         rpn_proposals = anno['rpn_proposals']
#         for i in range(4):
#             if len(old_lane[i]) < 2:
#                 pass
        # removing lanes with less than 2 points
        #print('&&', len(old_lanes))
#         if len(old_lanes) ==5:
#             print(old_lanes)
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        #print('***', len(old_lanes))
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        #print('****', len(old_lanes))
        # normalize the annotation coordinates  对于线上的点在这里进行resize处理，其它在get_item的transform里
        old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
                     for lane in old_lanes]
        # 对消失点与角度进行normalize CCTODO
        #print('*****', len(old_lanes))
        # create tranformed annotations  车道线上面的部分取-10000，对下面的部分进行补充
        lanes = np.ones((self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes[:4]):
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            #归一化的y
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs
        
        #增加rpn_proposals
        
        new_anno = {'path': anno['path'], 'label': lanes, 'rpn_proposals': anno['rpn_proposals'], 'vp_idx': anno['vp_idx'], 'old_anno': anno}
        return new_anno

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def label_to_lanes(self, label):
        lanes = []
        for l in label:
            if l[1] == 0:
                continue
            xs = l[5:] / self.img_w
            ys = self.offsets_ys / self.img_h
            start = int(round(l[2] * self.n_strips))
            length = int(round(l[4]))
            if length == 1:
                start = start - 1
                length = 2           
            xs = xs[start:start + length][::-1]
            ys = ys[start:start + length][::-1]
            
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            points = np.hstack((xs, ys))

            lanes.append(Lane(points=points))
        return lanes

    def draw_annotation(self, idx, label=None, pred=None, img=None, rpn_proposals = None, gt_vp = None, pred_vp=None):
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            img, label, _, _, _,_ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _, _, _,_  = self.__getitem__(idx)
            try:
                label = self.label_to_lanes(label)
            except:
                import pdb
                pdb.set_trace()
        img = cv2.resize(img, (self.img_w, self.img_h))

        img_h, _, _ = img.shape
        
        
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            # print(len(pred), 'preds')
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            # print('fp: {} | fn: {}'.format(fp, fn))
            # print(len(matches), 'matches')
            # print(matches, accs)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        else:
            fp = fn = None
        #画出rpn_proposals
#         import pdb
#         pdb.set_trace()
#         for line_i in range(4):
#             proposal = rpn_proposals[line_i]
#             end_point_y = proposal[1] + 200
#             end_point_x = (end_point_y - proposal[1]) / np.tan(proposal[3] * np.pi / 180)  + proposal[0]
#             cv2.line(img, tuple(proposal[:2].astype(np.int)), (int(round(end_point_x)), int(end_point_y )), (0, 255, 255), 2)
        for matches, accs, datum in data:
            for i, l in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                points = l.points
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int)
                points += pad
                xs, ys = points[:, 0], points[:, 1]
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.line(img,
                                   tuple(curr_p),
                                   tuple(next_p),
                                   color=color,
                                   thickness=2 if matches is None else 2)
                # if 'start_x' in l.metadata:
                #     start_x = l.metadata['start_x'] * img.shape[1]
                #     start_y = l.metadata['start_y'] * img.shape[0]
                #     cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                #                radius=5,
                #                color=(0, 0, 255),
                #                thickness=-1)
                # if len(xs) == 0:
                #     print("Empty pred")
                # if len(xs) > 0 and accs is not None:
                #     cv2.putText(img,
                #                 '{:.0f} ({})'.format(accs[i] * 100, i),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=color)
                #     cv2.putText(img,
                #                 '{:.0f}'.format(l.metadata['conf'] * 100),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=(255, 0, 255))
#         cv2.circle(img, tuple(gt_vp), 5, GT_COLOR, -1)
#         cv2.circle(img, tuple(pred_vp), 5, (255, 255, 255), -1)   
        cv2.circle(img, tuple(int(x) for x in gt_vp), 5, GT_COLOR, -1)
        cv2.circle(img, tuple(int(x) for x in pred_vp), 5, (255, 255, 255), -1)   
        img = cv2.resize(img, (self.dataset.img_w, self.dataset.img_h))
        return img, fp, fn

    def draw_annotation_point(self, idx, label=None, pred=None, img=None, rpn_proposals = None, gt_vp = None, pred_vp=None):
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            img, label, _, _, _,_ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _, _, _,_  = self.__getitem__(idx)
            try:
                label = self.label_to_lanes(label)
            except:
                import pdb
                pdb.set_trace()
        img = cv2.resize(img, (self.dataset.img_w, self.dataset.img_h))
        ori_img = img.copy()
        img_h, _, _ = img.shape
        
        
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            # print(len(pred), 'preds')
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            # print('fp: {} | fn: {}'.format(fp, fn))
            # print(len(matches), 'matches')
            # print(matches, accs)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        else:
            fp = fn = None
        for matches, accs, datum in data:
            for i, l in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                    continue
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                points = l.points
                points[:, 0] *= img.shape[1]
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int)
                points += pad
                xs, ys = points[:, 0], points[:, 1]
                for point in points:
                    img = cv2.circle(img, tuple(point), color=color, radius=5, thickness=-1)
        img = cv2.resize(img, (self.dataset.img_w, self.dataset.img_h))
        return ori_img, img, fp, fn

    
    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, idx):
        item = self.dataset[idx]
     
        print(item['path'])
        print(self.dataset.img_w,self.dataset.img_h)
        img_org = cv2.imread(item['path'])
        img_vp = cv2.imread(item['old_anno']['vp_lbpth'], cv2.IMREAD_GRAYSCALE)
        seg_img = cv2.imread(item['old_anno']['seg_lbpth'], cv2.IMREAD_GRAYSCALE)
        if self.dataset.__class__.__name__ == 'Ziyan':
            img_org = cv2.resize(img_org, (self.dataset.img_w, self.dataset.img_h))
            img_vp = cv2.resize(img_vp, (self.dataset.img_w, self.dataset.img_h))
            seg_img = cv2.resize(seg_img, (self.dataset.img_w, self.dataset.img_h))
        #seg_img = self.dataset.convert_labels(img_vp)
        img_vp = self.dataset.convert_vp_labels(img_vp)
        seg_img = self.dataset.convert_labels(seg_img)
        #print(item['path']) 
        line_strings_org = self.lane_to_linestrings(item['old_anno']['lanes'])
        #print('*', len(line_strings_org))
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        rpn_proposals = item['rpn_proposals']
        vp_idx = item['vp_idx']
        
        rpn_proposals = np.array(rpn_proposals)
        #print(item['old_anno']['seg_lbpth'])
        try:
            vp_lane_segmap = SegmentationMapsOnImage(np.stack((img_vp, seg_img), axis=2), shape=img_vp.shape)
        except:
            seg_img = cv2.resize(seg_img, (1920, 1080))
            vp_lane_segmap = SegmentationMapsOnImage(np.stack((img_vp, seg_img), axis=2), shape=img_vp.shape)
            print('1088')
        
        for i in range(30):
            
            img, line_strings, vp_lane_label = self.transform(image=img_org.copy(), line_strings=line_strings_org, segmentation_maps = vp_lane_segmap)
#             print('**', len(line_strings_org))
#             print('****', len(line_strings), line_strings)
            vp_label = vp_lane_label.arr[:, :, 0]
            lane_label = vp_lane_label.arr[:, :, 1]
            #因为transform 4条线成了5条, 一条弯线被截断，中间的一部分成了图片外面
            line_strings.clip_out_of_image_()
#             print('*****', len(line_strings), line_strings)
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings), 'rpn_proposals': item['rpn_proposals'], 'vp_idx' : item['vp_idx']}
            #print('***', len(self.linestrings_to_lanes(line_strings)))
            try:
                label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()
            #label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        vp_label = vp_label.astype(np.int64)
        lane_label = lane_label.astype(np.int64)
        return (img, label, np.array([0, 0]), np.array([0, 0]), vp_label, lane_label)


    def __len__(self):
        return len(self.dataset)
