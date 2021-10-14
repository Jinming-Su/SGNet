import os.path as osp
import os
import json
import cv2
import mmcv
import numpy as np
from torch.utils.data import Dataset
from mmdet.core import eval_map, eval_lane_recalls
from mmdet.core.bbox.bbox_dis import get_roi2align
 
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
from  tqdm import tqdm
import torch
import torch.nn.functional as F

from multiprocessing.dummy import Pool, Lock
# pool = Pool(processes=4)
# lock = Lock()
# dump_to_json = []
# global_cnt = 0
from mmdet.core.bbox.anchorandrect import *
@DATASETS.register_module()
class LaneDataset(CustomDataset):
    """Custom dataset for detection.
 
        The annotation format is shown as follows. The `ann` field is optional for
        testing.
 
        .. code-block:: none
 
            [
                {
                    'filename': 'a.jpg',
                    'width': 1280,
                    'height': 720,
                    'ann': {
                        'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                        'labels': <np.ndarray> (n, ),
                        'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                        'labels_ignore': <np.ndarray> (k, 4) (optional field)
                    }
                },
                ...
            ]
 
        Args:
            ann_file (str): Annotation file path.
            pipeline (list[dict]): Processing pipeline.
            classes (str | Sequence[str], optional): Specify classes to load.
                If is None, ``cls.CLASSES`` will be used. Default: None.
            data_root (str, optional): Data root for ``ann_file``,
                ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
            test_mode (bool, optional): If set True, annotation will not be loaded.
            filter_empty_gt (bool, optional): If set true, images without bounding
                boxes will be filtered out.
        """
    CLASSES = ('lane', )
    def __init__(self, ann_file, pipeline, img_dir, ann_dir, img_postfix='jpg', mask_label_dir=None, test_mode=False):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        #存储图片的txt的文件
        self.ann_file =ann_file
        self.post_fix = img_postfix
        self.mask_label_dir = mask_label_dir
        self.pipeline = pipeline
        self.test_mode = test_mode
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(ann_file)
        # set group flag for the sampler
        # processing pipeline
        self.pipeline = Compose(pipeline)
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        self.proposals = None
 
 
    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = '{}.{}'.format(img_id, self.post_fix )
            txt_path = osp.join(self.ann_dir, '{}.txt'.format(img_id))
            ann_content = mmcv.list_from_file(txt_path)
            line=[]
            bboxes = []
            labels = []
            if ann_content:
                line = ann_content[0].strip('\n').split(',')
                line = list(map(float, line))
                bboxes = []
                labels = []
                for bbox_anno in ann_content[1:]:
                    bbox = bbox_anno.strip('\n').split(',')
                    bbox = list(map(float, bbox))                  
                    bboxes.append(bbox[:3] + bbox[-1:])  #暂时不load h
                    labels.append(0)
            if not bboxes:
                line = np.zeros(2)
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0, ))
                masks = os.path.join(self.mask_label_dir , '{}.npy'.format(img_id))
            else:
                line = np.array(line)
                bboxes = np.array(bboxes)
                bboxes[:, 2][bboxes[:, 2] < 24.0] += 16.0
                labels = np.array(labels)
                masks = os.path.join(self.mask_label_dir , '{}.npy'.format(img_id))
            img_infos.append(dict(id=img_id, filename=filename, width=None, height=None, ann=dict(
                line=line.astype(np.float32) ,
                bboxes=bboxes.astype(np.float32),
                labels=labels.astype(np.int64),
                masks=masks
                
            )))
                # with open(txt_file, 'r+') as f:
                #     ann_contend = f.readlines()
                #     if not line:
                #         ann = None
                #     else:
                #         line = line.strip('\n').split(',')
                #         line = list(map(float, line))
                #         self.anno_list.append(line)
 
        return img_infos
        # for img_file in tqdm(self.img_list):
        #     frame_mp4_number = os.sep.join(img_file.split(os.sep)[-3:])[:-4]
        #     anno_file = os.path.join(self.cfg.anno_dir, frame_mp4_number + '.txt')
        #     with open(anno_file, 'r+') as f:
        #         line = f.readline()
        #         if not line:
        #             self.anno_list.append([-1, -1])
        #         else:
        #             line = line.strip('\n').split(',')
        #             line = list(map(float, line))
        #             self.anno_list.append(line)
    
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_dir
        results['bbox_fields'] = []
        results['line_fields'] = []
        results['mask_fields'] = []
    
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
 
        Args:
            idx (int): Index of data.
 
        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
 
    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.
 
        Args:
            idx (int): Index of data.
 
        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """
 
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.
 
        Args:
            idx (int): Index of data.
 
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(30, 50, 100),
                 iou_thr=(0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99),
                 scale_ranges=None):
        """Evaluate the dataset.
 
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        results_line = results[0]
        results_proposal = results[1]
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results_proposal,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_lane_recalls(
                gt_bboxes, results_proposal, proposal_nums, iou_thr, logger=logger, img_shape=(590, 1640))
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
    
    def find_low_recall(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=50,
                 iou_thr=0.95,
                 scale_ranges=None,
                 out_dir='./tmp/low_recall/'):
        """Evaluate the dataset.
 
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        results_line = results[0]
        results_proposal = results[1]
        print('proposal_nums {}, iou_thr {}, write to {}'.format(proposal_nums, iou_thr, out_dir))
        seg_label_dir = '/workdir/sujinming/0_lane_detection_paper/dataset/CULane/laneseg_label_w16/'
        img_dir = '/workdir/sujinming/0_lane_detection_paper/dataset/CULane/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        gt_bboxes = [ann['bboxes'] for ann in annotations]
        if isinstance(iou_thr, float):
            iou_thr = [iou_thr]
        low_recall_list = []
        # iou和proposal num设为1个
        for i in range(len(self)):
            recalls = eval_lane_recalls(
                gt_bboxes[i:i+1], results_proposal[i:i+1], proposal_nums, iou_thr, logger=logger, img_shape=(590, 1640), show_recall=False)
            if recalls[0][0] != 1 and (not np.isnan(recalls[0][0])):
                img_id = self.data_infos[i]['id']
                print('recalls', recalls, img_id)
                low_recall_list.append(i)
        print('len', len(low_recall_list))    
        for img_i in low_recall_list:
            img_id = self.data_infos[img_i]['id']
            result = results_proposal[img_i]
            pred_line = results_line[img_i]
            img_file = os.path.join(img_dir, img_id + '.jpg')
            img = cv2.imread(img_file)
            seg_img_file = os.path.join(seg_label_dir, img_id + '.png')
            img_id_2  = '_'.join(img_id.split(os.sep))
            seg_img = cv2.imread(seg_img_file) * 50
            anno_line = self.get_ann_info(img_i)['line']
            H = seg_img.shape[0]
            W = seg_img.shape[1]
            cv2.line(seg_img, (0, int(anno_line[0])), (W-1, int(anno_line[1])), (0, 255, 255), 2)
            cv2.line(seg_img, (0, int(H * pred_line[0])), (W-1, int(H * pred_line[1])), (255, 0, 0), 2)
            #result = result[result[:, 4]>score_thr]
            sort_idx = np.argsort(result[:, 4])[::-1][:proposal_nums]
            result = result[sort_idx]
            result_rects = anchor2rect_mask_np(result)
            for rect_i, rect in enumerate(result_rects): 
                score = result[rect_i][4]
                draw_rot_rect(rect, seg_img)
                cv2.putText(seg_img, str(score), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            stack_img = np.vstack((seg_img, img))
            cv2.imwrite(os.path.join(out_dir ,  img_id_2 + '.jpg'), stack_img) 
#         if isinstance(iou_thr, float):
#             iou_thr = [iou_thr]
        
#         if metric == 'mAP':
#             assert isinstance(iou_thr, float)
#             mean_ap, _ = eval_map(
#                 results,
#                 annotations,
#                 scale_ranges=scale_ranges,
#                 iou_thr=iou_thr,
#                 dataset=self.CLASSES,
#                 logger=logger)
#             eval_results['mAP'] = mean_ap
#         elif metric == 'recall':
#             gt_bboxes = [ann['bboxes'] for ann in annotations]
#             if isinstance(iou_thr, float):
#                 iou_thr = [iou_thr]
#             recalls = eval_lane_recalls(
#                 gt_bboxes, results, proposal_nums, iou_thr, logger=logger, img_shape=(590, 1640))
#             for i, num in enumerate(proposal_nums):
#                 for j, iou in enumerate(iou_thr):
#                     eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
#             if recalls.shape[1] > 1:
#                 ar = recalls.mean(axis=1)
#                 for i, num in enumerate(proposal_nums):
#                     eval_results[f'AR@{num}'] = ar[i]
#         return eval_results
    
    def visualize_rpn(self,
                 results,
                 logger=None,
                 proposal_nums=200,
                 score_thr=0.0,
                 scale_ranges=None,
                 out_dir = '/workdir/chenchao/projects/mmdetection-master/result_vis/'):
        results_line = results[0]
        results_proposal = results[1]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('score_thr {}, proposal_nums {}, write to {}'.format(score_thr, proposal_nums, out_dir))
        seg_label_dir = '/home/hadoop-mtcv/cephfs/data/chenchao60/CULane/laneseg_label_w16/'
        img_dir = '/workdir/chenchao/data/CULane/'
        assert len(results_proposal) == len(self)
        for img_i in tqdm(range(len(self))):
            result = results_proposal[img_i]
            pred_line = results_line[img_i]
            img_id = self.data_infos[img_i]['id']
            img_file = os.path.join(img_dir, img_id + '.jpg')
            img = cv2.imread(img_file)
            seg_img_file = os.path.join(seg_label_dir, img_id + '.png')
            anno_line = self.get_ann_info(img_i)['line']
            img_id_2  = '_'.join(img_id.split(os.sep))
            seg_img = cv2.imread(seg_img_file) * 50
            H = seg_img.shape[0]
            W = seg_img.shape[1]
            cv2.line(seg_img, (0, int(anno_line[0])), (W-1, int(anno_line[1])), (0, 255, 255), 2)
            cv2.line(seg_img, (0, int(H * pred_line[0])), (W-1, int(H * pred_line[1])), (255, 0, 0), 2)
            #result = result[result[:, 4]>score_thr]
            sort_idx = np.argsort(result[:, 4])[::-1][:proposal_nums]
            result = result[sort_idx]
            result_rects = anchor2rect_mask_np(result)
            for rect_i, rect in enumerate(result_rects): 
                score = result[rect_i][4]
                draw_rot_rect(rect, seg_img)
                cv2.putText(seg_img, str(score), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            stack_img = np.vstack((seg_img, img))
            cv2.imwrite(os.path.join(out_dir ,  img_id_2 + '.jpg'), stack_img) 
    
    def visualize_rcnn(self,
                 results,
                 logger=None,
                 proposal_nums=200,
                 score_thr=0.0,
                 scale_ranges=None,
                 out_dir = ''):
 
        # results[0]: line list
        # results[1]: proposal and mask list
        # results[1][i][0][0]: proposal
        # results[1][i][1][0][0]: mask
 
        results_line = results[0]
        results_proposal = results[1]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('score_thr {}, proposal_nums {}, write to {}'.format(score_thr, proposal_nums, out_dir))
        seg_label_dir = '/workdir/chenchao/projects/mmdetection-master/data/Tusimple/testing/gt_instance_image/'
        img_dir = '/workdir/chenchao/projects/mmdetection-master/data/Tusimple/testing/gt_image/'
        assert len(results_proposal) == len(self)
        dump_to_json = []
        for img_i in range(len(self)):
            result = results_proposal[img_i][0]
            pred_line = results_line[img_i]
            img_id = self.data_infos[img_i]['id']
            img_file = os.path.join(img_dir, img_id + '.jpg')
            img = cv2.imread(img_file)
            seg_img_file = os.path.join(seg_label_dir, img_id + '.png')
 
            anno_line = self.get_ann_info(img_i)['line']
            img_id_2  = '_'.join(img_id.split(os.sep))
            seg_img = cv2.imread(seg_img_file) * 50
 
            H = seg_img.shape[0]
            W = seg_img.shape[1]
            cv2.line(seg_img, (0, int(anno_line[0])), (W-1, int(anno_line[1])), (0, 255, 255), 2)
            cv2.line(seg_img, (0, int(H * pred_line[0])), (W-1, int(H * pred_line[1])), (255, 0, 0), 2)
            result = result[0][result[0][:, 4]>score_thr]
            sort_idx = np.argsort(result[:, 4])[::-1][:proposal_nums]
            result = result[sort_idx]
            result_rects = anchor2rect_mask_np(result)
 
 
            mask_img = np.zeros((seg_img.shape[0], seg_img.shape[1]))
            if len(result_rects) > 0:
                mask_ori = results[1][img_i][1].cuda()
                bbox_ori = torch.from_numpy(results[1][img_i][0][0][:, :-1]).float().cuda()
                det_labels = torch.zeros((mask_ori.shape[0])).long().cuda()
                rcnn_test_cfg_mask_thr_binary = 0.5
                ori_shape = seg_img.shape
                scale_factor = torch.from_numpy(np.array([1., 1., 1., 1.])).float().cuda()
                rescale = True
                result_masks = self.get_seg_masks(mask_ori, bbox_ori, det_labels, rcnn_test_cfg_mask_thr_binary, ori_shape, scale_factor, rescale)
 
                #result_masks = results_proposal[img_i][1][0]
                result_masks = np.array(result_masks)[0][sort_idx]
                for mask_i, mask in enumerate(result_masks):
                    mask_img[mask] = int(255 / (len(result_masks) + 1) * (mask_i + 1))
 
            # 预测结果转化为左2、左1、右1、右2的格式。值为一个形状为(5, H, W)的图像
            maxnum_lane = 0
            row_maxnum_lane = 0
            for index_row in range(0, mask_img.shape[0], int(mask_img.shape[0]/20)):
                eachnum_lane = len(np.unique(mask_img[index_row])) - 1
                if eachnum_lane >= maxnum_lane:
                    maxnum_lane = eachnum_lane
                    row_maxnum_lane = index_row
            zuo2 = set()
            zuo1 = set()
            you1 = set()
            you2 = set()
 
            index_start_left = int(mask_img.shape[1]/2)
            index_start_right = int(mask_img.shape[1]/2)
            if mask_img[row_maxnum_lane, int(mask_img.shape[1]/2)] > 0:
                while (index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] > 0):
                    zuo1.add(mask_img[row_maxnum_lane, index_start_left])
                    index_start_left -= 1
                while (index_start_right <= mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] > 0):
                    you1.add(mask_img[row_maxnum_lane, index_start_right])
                    index_start_right += 1
 
                if mask_img.shape[1] - index_start_left >= index_start_right - index_start_right:
                    you1.clear()
                else:
                    zuo1.clear()
 
            if mask_img[row_maxnum_lane, index_start_left] == 0:
                while (index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] == 0): 
                    index_start_left -= 1
                if len(zuo1) == 0:
                    while(index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] > 0):
                        zuo1.add(mask_img[row_maxnum_lane, index_start_left])
                        index_start_left -= 1
                while (index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] == 0):
                    index_start_left -= 1
                if len(zuo2) == 0:
                    while(index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] > 0):
                        zuo2.add(mask_img[row_maxnum_lane, index_start_left])
                        index_start_left -= 1
            if mask_img[row_maxnum_lane, index_start_right] == 0:
                while (index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] == 0):
                    index_start_right += 1
                if len(you1) == 0:
                    while(index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] > 0):
                        you1.add(mask_img[row_maxnum_lane, index_start_right])
                        index_start_right += 1
                while (index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] == 0):
                    index_start_right += 1
                if len(you2) == 0:
                    while(index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] > 0):
                        you2.add(mask_img[row_maxnum_lane, index_start_right])
                        index_start_right += 1
            res_mat = np.zeros((mask_img.shape[0], mask_img.shape[1], 5))
            for idx in zuo2:
                res_mat[:, :, 1][mask_img == idx] = 1
            for idx in zuo1:
                res_mat[:, :, 2][mask_img == idx] = 1
            for idx in you1:
                res_mat[:, :, 3][mask_img == idx] = 1
            for idx in you2:
                res_mat[:, :, 4][mask_img == idx] = 1
            res_mat[:, :, 0] = 1- res_mat[:, :, 1] - res_mat[:, :, 2] - res_mat[:, :, 3] - res_mat[:, :, 4]
 
            import sys
            sys.path.append('./evaluateion/prob2lines')
            from getLane import prob2lines_CULane
            res_list = prob2lines_CULane(res_mat.transpose((2, 0, 1)), [int(len(zuo2) > 0), int(len(zuo1) > 0), int(len(you1) > 0), int(len(you2) > 0)])
            output_dir = out_dir + '/output/' + os.path.dirname(img_id)
            os.makedirs(output_dir, exist_ok = True)
            with open(output_dir + os.sep + os.path.basename(img_id) + '.lines.txt', 'w+') as fw:
                for each_line in res_list:
                    for point in each_line:
                        fw.write('{} {} '.format(point[0], point[1]))
                    fw.write('\n')
 
            for rect_i, rect in enumerate(result_rects): 
                score = result[rect_i][4]
                draw_rot_rect(rect, seg_img)
                cv2.putText(seg_img, str(score), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            stack_img = np.vstack((np.repeat(mask_img[:, :, np.newaxis], repeats=3, axis=-1), seg_img, img))
            cv2.imwrite(os.path.join(out_dir ,  img_id_2 + '.jpg'), stack_img)
     
    
    def visualize_rcnn_tusimple(self,
                 results,
                 logger=None,
                 proposal_nums=200,
                 score_thr=0.0,
                 scale_ranges=None,
                 out_dir = ''):
 
        # results[0]: line list
        # results[1]: proposal and mask list
        # results[1][i][0][0]: proposal
        # results[1][i][1][0][0]: mask
        with open('/workdir/chenchao/projects/mmdetection-master/data/Tusimple/testing/test_for_eval.txt', 'r') as f:
            img_list_tusimple = f.readlines()
        results_line = results[0]
        results_proposal = results[1]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('score_thr {}, proposal_nums {}, write to {}'.format(score_thr, proposal_nums, out_dir))
        seg_label_dir = '/workdir/chenchao/projects/mmdetection-master/data/Tusimple/testing/gt_instance_image/'
        img_dir = '/workdir/chenchao/projects/mmdetection-master/data/Tusimple/testing/gt_image/'
        assert len(results_proposal) == len(self)
        dump_to_json = []
        for img_i in range(len(self)):
            result = results_proposal[img_i][0]
            pred_line = results_line[img_i]
            img_id = self.data_infos[img_i]['id']
            img_file = os.path.join(img_dir, img_id + '.png')
            img = cv2.imread(img_file)
            seg_img_file = os.path.join(seg_label_dir, img_id + '.png')
 
            anno_line = self.get_ann_info(img_i)['line']
            img_id_2  = '_'.join(img_id.split(os.sep))
            seg_img = cv2.imread(seg_img_file) * 50
 
            H = seg_img.shape[0]
            W = seg_img.shape[1]
            cv2.line(seg_img, (0, int(anno_line[0])), (W-1, int(anno_line[1])), (0, 255, 255), 2)
            cv2.line(seg_img, (0, int(H * pred_line[0])), (W-1, int(H * pred_line[1])), (255, 0, 0), 2)
            result = result[0][result[0][:, 4]>score_thr]
            sort_idx = np.argsort(result[:, 4])[::-1][:proposal_nums]
            result = result[sort_idx]
            result_rects = anchor2rect_mask_np(result)
 
 
            mask_img = np.zeros((seg_img.shape[0], seg_img.shape[1]))
            if len(result_rects) > 0:
                mask_ori = results[1][img_i][1].cuda()
                bbox_ori = torch.from_numpy(results[1][img_i][0][0][:, :-1]).float().cuda()
                det_labels = torch.zeros((mask_ori.shape[0])).long().cuda()
                rcnn_test_cfg_mask_thr_binary = 0.5
                ori_shape = seg_img.shape
                scale_factor = torch.from_numpy(np.array([1., 1., 1., 1.])).float().cuda()
                rescale = True
                result_masks = self.get_seg_masks(mask_ori, bbox_ori, det_labels, rcnn_test_cfg_mask_thr_binary, ori_shape, scale_factor, rescale)
 
                #result_masks = results_proposal[img_i][1][0]
                result_masks = np.array(result_masks)[0][sort_idx]
                for mask_i, mask in enumerate(result_masks):
                    mask_img[mask] = int(255 / (len(result_masks) + 1) * (mask_i + 1))
 
            # 预测结果转化为左2、左1、右1、右2的格式。值为一个形状为(5, H, W)的图像
            maxnum_lane = 0
            row_maxnum_lane = 0
            for index_row in range(0, mask_img.shape[0], int(mask_img.shape[0]/20)):
                eachnum_lane = len(np.unique(mask_img[index_row])) - 1
                if eachnum_lane >= maxnum_lane:
                    maxnum_lane = eachnum_lane
                    row_maxnum_lane = index_row
            zuo2 = set()
            zuo1 = set()
            you1 = set()
            you2 = set()
 
            index_start_left = int(mask_img.shape[1]/2)
            index_start_right = int(mask_img.shape[1]/2)
            if mask_img[row_maxnum_lane, int(mask_img.shape[1]/2)] > 0:
                while (index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] > 0):
                    zuo1.add(mask_img[row_maxnum_lane, index_start_left])
                    index_start_left -= 1
                while (index_start_right <= mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] > 0):
                    you1.add(mask_img[row_maxnum_lane, index_start_right])
                    index_start_right += 1
 
                if mask_img.shape[1] - index_start_left >= index_start_right - index_start_right:
                    you1.clear()
                else:
                    zuo1.clear()
 
            if mask_img[row_maxnum_lane, index_start_left] == 0:
                while (index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] == 0): 
                    index_start_left -= 1
                if len(zuo1) == 0:
                    while(index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] > 0):
                        zuo1.add(mask_img[row_maxnum_lane, index_start_left])
                        index_start_left -= 1
                while (index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] == 0):
                    index_start_left -= 1
                if len(zuo2) == 0:
                    while(index_start_left >= 0 and mask_img[row_maxnum_lane, index_start_left] > 0):
                        zuo2.add(mask_img[row_maxnum_lane, index_start_left])
                        index_start_left -= 1
            if mask_img[row_maxnum_lane, index_start_right] == 0:
                while (index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] == 0):
                    index_start_right += 1
                if len(you1) == 0:
                    while(index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] > 0):
                        you1.add(mask_img[row_maxnum_lane, index_start_right])
                        index_start_right += 1
                while (index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] == 0):
                    index_start_right += 1
                if len(you2) == 0:
                    while(index_start_right < mask_img.shape[1] and mask_img[row_maxnum_lane, index_start_right] > 0):
                        you2.add(mask_img[row_maxnum_lane, index_start_right])
                        index_start_right += 1
            res_mat = np.zeros((mask_img.shape[0], mask_img.shape[1], 5))
            for idx in zuo2:
                res_mat[:, :, 1][mask_img == idx] = 1
            for idx in zuo1:
                res_mat[:, :, 2][mask_img == idx] = 1
            for idx in you1:
                res_mat[:, :, 3][mask_img == idx] = 1
            for idx in you2:
                res_mat[:, :, 4][mask_img == idx] = 1
            res_mat[:, :, 0] = 1- res_mat[:, :, 1] - res_mat[:, :, 2] - res_mat[:, :, 3] - res_mat[:, :, 4]
            
            
            
            img_tusimple = img_list_tusimple[int(img_id)].strip('\n')
            import sys
            sys.path.append('./evaluateion/prob2lines')
#             import pdb
#             pdb.set_trace()
            from getLane import prob2lines_CULane,  polyfit2coords_tusimple, polyfit2coords_tusimple_gai
#             import pdb
#             pdb.set_trace()
            lane_coords = polyfit2coords_tusimple_gai(res_mat, crop_h=0, resize_shape=(720, 1280), y_px_gap=10, pts=56, ord=2)
            #if lane_coords == []:
                
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])
            #path_tree = split_path(img_name[b])
#             save_dir, save_name = path_tree[-3:-1], path_tree[-1]
#             save_dir = os.path.join(out_path, *save_dir)
#             save_name = save_name[:-3] + "lines.txt"
#             save_name = os.path.join(save_dir, save_name)
            save_dir, save_name = './tusimple_eval/', './tusimple_eval/{}lines.txt'.format(img_id)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)

            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = img_tusimple
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            if len(lane_coords) != 0:
                for (x, y) in lane_coords[0]:
                    json_dict['h_sample'].append(y)
            dump_to_json.append(json.dumps(json_dict))  
            
            
            
            mask_img = np.repeat(mask_img[:, :, np.newaxis], repeats=3, axis=-1)
            #mask_img_zeros = np.repeat(mask_img_zeros[:, :, np.newaxis], repeats=3, axis=-1)
            for rect_i, rect in enumerate(result_rects): 
                score = result[rect_i][4]
                #draw_rot_rect(rect, mask_img_zeros)
                draw_rot_rect(rect, seg_img)
                cv2.putText(seg_img, str(score), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
            stack_img = np.vstack((mask_img,  seg_img, img))
            cv2.imwrite(os.path.join(out_dir ,  img_id_2 + '.jpg'), stack_img)
        
        
        with open(os.path.join(out_dir, "predict_test.json"), "w") as f:
            for line in dump_to_json:
                print(line, end="\n", file=f)
        
    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg_mask_thr_binary,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.
 
        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size
 
        Returns:
            list[list]: encoded masks
        """
        self.num_classes = 1
        self.class_agnostic = False
        BYTES_PER_FLOAT = 4
        GPU_MEM_LIMIT = 1024**3
 
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)
 
        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels
 
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
            scale_factor = 1.0
 
        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor
 
        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
 
        threshold = rcnn_test_cfg_mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)
 
        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]
        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')
 
            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)
 
            im_mask[(inds,) + spatial_inds] = masks_chunk
 
        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].cpu().numpy())
        return cls_segms
 
def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.
    修改为斜框的paste_masks

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    import time
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    start = time.time()
    boxes2align = get_roi2align(boxes, img_h, img_w).unsqueeze(-1)
    x_center, y_center, w, h, theta = torch.split(boxes2align, 1, dim=1)
    theta = (theta) * np.pi / 180.0

    N = masks.shape[0]
    # start = time.time()
    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32)
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32)
    img_x = img_x[None, None, :].expand(N, img_y.size(0), img_x.size(0))
    img_y = img_y[None, :, None].expand(N, img_y.size(0), img_x.size(2))
    # print('expand', time.time() - start)
    # start = time.time()
    w_gap = (img_x - x_center) * torch.sin(theta) + (img_y - y_center) * torch.cos(theta)
    h_gap = (img_x - x_center) * torch.cos(theta) - (img_y - y_center) * torch.sin(theta)
    norm_y = w_gap / w * 2
    norm_x = h_gap / h * 2
    grid = torch.stack([norm_x, norm_y], dim=3)
    # print('gap', time.time() - start)
    start = time.time()
    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)
    # print('F.grid_sample', time.time() - start)
    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
