import os
import json
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
#import mmcv
import utils.culane_metric as culane_metric
import cv2
from .lane_dataset_loader import LaneDatasetLoader

SPLIT_FILES = {
    'train': "list/train.txt",
    'val': 'list/test_random_1000.txt',
    'test': "list/test.txt",
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
    'debug': 'list/test_random_1000.txt',
    'test_random_100': 'list/test_random_100.txt',
    'test_1': 'list/test_random_1.txt',
    'test_random_1000': 'list/test_random_1000.txt',
    'test_random_10000': 'list/test_random_10000.txt',
    'debug_5lane': 'list/debug_5lane.txt',
    'train_1000': 'list/train_random_1000.txt',
    'test_1000_tr': 'list/train_random_1000.txt',
    'test_100': 'list/test_random_100.txt',
    'test_1000': 'list/test_random_1000.txt',
    'train_10000': 'list/train_random_1W.txt',
    'train_1000_from_line': 'list/test_random_1000.txt',
    'train_10000_from_line' : 'list/train_random_1W.txt',
    'train_from_line': "list/train.txt",
}


class CULane(LaneDatasetLoader):
    def __init__(self, max_lanes=None, split='train', root=None, official_metric=True, list_root='/home/hadoop-mtcv/cephfs/data/chenchao60/0_lane_det_code/CULane_txt'):
        #
        #split = 'test_1000'
        #split = 'train_1000'
        print('CULane split ****' , split)
        self.split = split
        self.root = root
        self.official_metric = official_metric
        self.logger = logging.getLogger(__name__)
        self.vp_path = './datasets/culane/vp_label_new_32'
        #self.vp_path = './datasets/culane/vp_label_new_from_line_32'
        self.lane_seg_path = './datasets/culane/laneseg_label_w16'
        if root is None:
            raise Exception('Please specify the root directory')
        if split not in SPLIT_FILES:
            raise Exception('Split `{}` does not exist.'.format(split))

        self.list = os.path.join(list_root, SPLIT_FILES[split])

        self.img_w, self.img_h = 1640, 590
        self.annotations = []
        self.rpn_label_dir = os.path.join(list_root, 'rpn_label_new')
        with open('./datasets/vp.json', 'r') as fr:
            labels_info = json.load(fr)
        self.vplb_map = {el['id']: el['trainId'] for el in labels_info}
        with open('./datasets/lane.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        self.load_annotations()
        self.max_lanes = 4 if max_lanes is None else max_lanes

    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def get_metrics(self, raw_lanes, idx):
        lanes = []
        pred_str = self.get_prediction_string(raw_lanes)
        for lane in pred_str.split('\n'):
            if lane == '':
                continue
            lane = list(map(float, lane.split()))
            lane = [(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
            lanes.append(lane)
        anno = culane_metric.load_culane_img_data(self.annotations[idx]['path'].replace('.jpg', '.lines.txt'))
        _, fp, fn, ious, matches = culane_metric.culane_metric(lanes, anno)

        return fp, fn, matches, ious

    def load_annotation(self, file):
        img_path = os.path.join(self.root, file)
        #print(img_path)
        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        
        #增加读取rpn的gt proposal
        rpn_label_pth = os.path.join(self.rpn_label_dir, file.replace('jpg', 'txt'))      
        # with open(rpn_label_pth) as f:
        #     ann_content = f.readlines()
        ann_content = None
        #ann_content = mmcv.list_from_file(rpn_label_pth) 
        bboxes = []
        if ann_content:
            line = ann_content[0].strip('\n').split(',')
            line = list(map(float, line))
            for bbox_anno in ann_content[1:]:
                bbox = bbox_anno.strip('\n').split(',')
                bbox = list(map(float, bbox))
                
                bboxes.append(bbox[:3] + bbox[-1:])  #暂时不load h
        
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        #print('*', len(data))
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        #print('**', len(lanes))
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points

            
        lanes = [lane for lane in lanes if len(lane) >= 2]  # remove lanes with less than 2 points
        
#         lanes_bak = lanes.copy()
#         bboxes_bak = bboxes.copy()
#         bboxes = []
#         lanes = []
#         for i, lane in enumerate(lanes_bak):
#             if len(lane) >= 2:
#                 lanes.append(lane)
#                 #处理unmatch问题
#                 try:
#                     bboxes.append(bboxes_bak[i])
#                 except:
#                     break
                      
        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        
       
        
        if not bboxes:
            bboxes =  -1e6  * np.ones((4, 4))             
        else:          
            bboxes = np.array(bboxes)
            bboxes = np.concatenate((bboxes, -1e6 * np.ones((4-len(bboxes), 4))))
        #json不能保存numpy数组
        bboxes = bboxes.tolist()
        vp_lbpth = os.path.join(self.vp_path, file.replace('jpg', 'png'))
        seg_lbpath = os.path.join(self.lane_seg_path, file.replace('jpg', 'png'))
        vp_label = Image.fromarray(cv2.imread(vp_lbpth, -1)).convert('L')
        vp_label = np.array(vp_label).astype(np.int64)[np.newaxis, :]
        vp_label= self.convert_labels(vp_label)
        vp_label_idx = np.where(vp_label[0] == 1)
        if len(vp_label_idx[0]) == 0:
#             labelvp_idx_x = 0
#             labelvp_idx_y = 0
            vp_label_idx = -1e6 * np.ones((2), dtype=np.float32)
        else:
            labelvp_idx_x = np.mean(vp_label_idx[1])
            labelvp_idx_y = np.mean(vp_label_idx[0])
            vp_label_idx = np.array([labelvp_idx_x, labelvp_idx_y]).astype(np.float32)
        vp_label_idx = vp_label_idx.tolist()
        #lanes 车道线条数，点的数目，2  的list[list[tuple
        return {'path': img_path, 'lanes': lanes, 'rpn_proposals': bboxes, 'vp_idx': vp_label_idx, 'vp_lbpth': vp_lbpth, 'seg_lbpth': seg_lbpath}

    def load_annotations(self):
        self.annotations = []
        self.max_lanes = 0
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/culane_{}.json'.format(self.split)

        if os.path.exists(cache_path):
            self.logger.info('Loading CULane annotations (cached)...')
            with open(cache_path, 'r') as cache_file:
                data = json.load(cache_file)
                self.annotations = data['annotations']
                self.max_lanes = data['max_lanes']
        else:
            self.logger.info('Loading CULane annotations and caching...')
            with open(self.list, 'r') as list_file:
                files = [line.rstrip()[1 if line[0] == '/' else 0::]
                         for line in list_file]  # remove `/` from beginning if needed

            for file in tqdm(files):
                img_path = os.path.join(self.root, file)
                anno = self.load_annotation(file)
                anno['org_path'] = file

                if len(anno['lanes']) > 0:
                    self.max_lanes = max(self.max_lanes, len(anno['lanes']))
                self.annotations.append(anno)
            with open(cache_path, 'w') as cache_file:
                json.dump({'annotations': self.annotations, 'max_lanes': self.max_lanes}, cache_file)

        self.logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.annotations),
                         self.max_lanes)

    def get_prediction_string(self, pred):
        ys = np.arange(self.img_h) / self.img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def eval_predictions(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.annotations[idx]['old_anno']['org_path']))
            output_filename = os.path.basename(self.annotations[idx]['old_anno']['org_path'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        return culane_metric.eval_predictions(output_basedir, self.root, self.list, official=self.official_metric)

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))
    
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label
    
    def convert_vp_labels(self, label):
        for k, v in self.vplb_map.items():
            label[label == k] = v
        return label
    
    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
