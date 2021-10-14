import pickle
import random
import logging

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
import pdb
from time import *
import torch.nn.functional as F

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

class Runner:
    def __init__(self, cfg, exp, device, args, resume=False, view=None, deterministic=False, mode=None):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.n_classes = 2
        self.ignore_label = 255
        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])
        self.mode = mode
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        #print('**************')
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader(self.mode)
        loss_parameters = self.cfg.get_loss_parameters()
        
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            #begin_time = time()
            for i, (images, labels, _, _, vp_labels, lane_labels) in enumerate(pbar):
                #print('data耗时',  time() - begin_time)
                #begin_time = time()
                images = images.to(self.device)
                labels = labels.to(self.device)
                vp_labels = vp_labels.to(self.device) 
                lane_labels = lane_labels.to(self.device)
                        #加上扰动
#                 for batch_idx in range(len(images)):          
#                     rpn_proposals[batch_idx][:, 0] += (torch.rand_like(rpn_proposals[batch_idx][:, 0]) * 40 - 20)
#                     rpn_proposals[batch_idx][:, 3] += (torch.rand_like(rpn_proposals[batch_idx][:, 3]) * 20 - 10)
#                     inter_p, _, _, _ = get_inter_with_border_mul(rpn_proposals[batch_idx], 295, 820)
#                     rpn_proposals[batch_idx] = torch.cat((inter_p, rpn_proposals[batch_idx][:, 2:]), dim=1)
                #print('加扰动耗时')
                # Forward pass
                outputs, vp_logits, pred_vps, lane_logits, h_para = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, vp_logits, pred_vps, labels, vp_labels, lane_logits, lane_labels, h_para,  **loss_parameters)
                #print('forward耗时', time() - begin_time)
                #begin_time = time()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()
                #print('backward耗时', time() - begin_time)
                #begin_time = time()
                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
                #end_time = time()
                #print('log耗时', time() - begin_time)
#                 except:
#                     import pdb
#                     pdb.set_trace()
#                     print('rpn_proposals', rpn_proposals)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=False)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch), strict=False)
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        vis_path = os.path.join('experiments', self.exp.name, 'vis_{}'.format(self.view))
        os.makedirs(vis_path, exist_ok=True)
        loss_avg = []
        loss_parameters = self.cfg.get_loss_parameters()
        ori_w =  dataloader.dataset.dataset.img_w
        ori_h = dataloader.dataset.dataset.img_h
        with torch.no_grad():
            for idx, (images, labels, rpn_proposals, vp_idx, vp_labels, lane_labels) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                labels = labels.to(self.device)
                vp_labels = vp_labels.to(self.device) 
                lane_labels = lane_labels.to(self.device)
                vp_idxs = []
                for vp_label in vp_labels:
                    vp_x_y = torch.mean(torch.nonzero(vp_label, as_tuple=False).float(), 0)
                    vp_idxs.append(vp_x_y)
                vp_idxs = torch.stack(vp_idxs)[:, [1, 0]]
                #预测出没有消失点的分割图，消失点设为0
                vp_idxs = torch.where(torch.isnan(vp_idxs), torch.tensor(-1e6, device = self.device), vp_idxs)   
                outputs, vp_logits, pred_vps, lane_logits, h_para = model(images, **test_parameters)
                try:
                    loss, _ = model.loss(outputs, vp_logits, pred_vps, labels, vp_labels, lane_logits, lane_labels, h_para,  **loss_parameters)
                    loss_avg.append(loss)
                except:
                    print('pass1')
                    pass
                try:
                    prediction = model.decode(outputs, as_lanes=True)
                    predictions.extend(prediction)
                except:
                    print('pass2')
                    pass
                
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0], gt_vp= vp_idxs[0].cpu().numpy(), pred_vp = pred_vps[0].cpu().numpy())
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue
                    if self.view == 'corrects' and fp !=0 and fn != 0:
                        continue
                    cv2.imwrite(os.path.join(vis_path, str(idx)+ '.jpg'), img)
                    if  self.view == 'corrects':
                        import pdb
                        pdb.set_trace()
            
                        vp_probs = F.softmax(vp_logits[0], 0)
                        vp_preds = torch.argmax(vp_probs, 0)
                        vp_map = vp_preds.cpu().numpy().astype(np.uint8)
#                         import pdb
#                         pdb.set_trace()
                        vp_map = cv2.resize(vp_map, (ori_w, ori_h))
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'vp')),  vp_map * 200)
                        seg_probs = F.softmax(lane_logits[0], 0)
                        seg_preds = torch.argmax(seg_probs, 0)
                        seg_map = seg_preds.cpu().numpy().astype(np.uint8)
                        seg_map = cv2.resize(seg_map, (ori_w, ori_h))
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'lane')), seg_map * 200)
    #                     cv2.imshow('pred', img)
    #                     cv2.waitKey(0)
        try:
            print('loss_avg', sum(loss_avg) / len(loss_avg))
        except:
            pass
        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)
        
        
    def draw(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch), strict=False)
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        for channel in range(1028, 1029):
            vis_path = os.path.join('experiments', self.exp.name, 'vis_{}_{}'.format(self.view, 'draw'))
            os.makedirs(vis_path, exist_ok=True)
            loss_avg = []
            loss_parameters = self.cfg.get_loss_parameters()
            ori_w =  dataloader.dataset.dataset.img_w
            ori_h = dataloader.dataset.dataset.img_h
            with torch.no_grad():
                for idx, (images, labels, rpn_proposals, vp_idx, vp_labels, lane_labels) in enumerate(tqdm(dataloader)):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    vp_labels = vp_labels.to(self.device) 
                    lane_labels = lane_labels.to(self.device)
                    lane_labels_copy  = lane_labels.detach().clone()
                    vp_idxs = []
                    for vp_label in vp_labels:
                        vp_x_y = torch.mean(torch.nonzero(vp_label, as_tuple=False).float(), 0)
                        vp_idxs.append(vp_x_y)
                    vp_idxs = torch.stack(vp_idxs)[:, [1, 0]]
                    #预测出没有消失点的分割图，消失点设为0
                    vp_idxs = torch.where(torch.isnan(vp_idxs), torch.tensor(-1e6, device = self.device), vp_idxs)

                    outputs, vp_logits, pred_vps, lane_logits, h_para, heatmap , heatmap_add= model.draw(images,  idx = idx, vis_path = vis_path, channel= channel, **test_parameters)
                    try:
                        loss, _ = model.loss(outputs, vp_logits, pred_vps, labels, vp_labels, lane_logits, lane_labels_copy, h_para,  **loss_parameters)
                        loss_avg.append(loss)
                    except:
                        print('pass1')
                        pass
                    try:
                        prediction = model.decode(outputs, as_lanes=True)
                        predictions.extend(prediction)
                    except:
                        print('pass2')
                        pass

                    if self.view:
                        img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        ori_img, img, fp, fn = dataloader.dataset.draw_annotation_point(idx, img=img, pred=prediction[0], gt_vp= vp_idxs[0].cpu().numpy(), pred_vp = pred_vps[0].cpu().numpy())
                        if self.view == 'mistakes' and fp == 0 and fn == 0:
                            continue
                        if self.view == 'corrects' and (fp !=0 or fn != 0):
                            continue
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'ori')), ori_img)
                        cv2.imwrite(os.path.join(vis_path, str(idx)+ '.jpg'), img)        
                        vp_probs = F.softmax(vp_logits[0], 0)
                        vp_preds = torch.argmax(vp_probs, 0)
                        vp_map = vp_preds.cpu().numpy().astype(np.uint8)
    #                         import pdb
    #                         pdb.set_trace()
                        vp_map = cv2.resize(vp_map, (ori_w, ori_h))
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'vp')),  vp_map * 255)
    #                         seg_probs = F.softmax(lane_logits[0], 0)
    #                         seg_preds = torch.argmax(seg_probs, 0)
    #                         seg_map = seg_preds.cpu().numpy().astype(np.uint8)
    #                         seg_map = cv2.resize(seg_map, (ori_w, ori_h))
    #                         cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'lane')), seg_map * 200)
                        pred_seg_dir = '/home/hadoop-mtcv/cephfs/data/chenchao60/0_lane_det_code/deeplabv3p_resnet50_lane/output'
                        seg_map = cv2.imread(os.path.join(pred_seg_dir, '{}_pred.png'.format(str(idx))))
                        seg_map = cv2.resize(seg_map, (ori_w, ori_h))
#                         image = images[0].cpu().numpy().astype(np.uint8)
#                         image = image.transpose(1, 2, 0)
#                         image = cv2.resize(image, (3, 1640, 590))
                        
                        lane_label = lane_labels[0].cpu().numpy().astype(np.uint8)
                        lane_label  = cv2.resize(lane_label , (1640, 590))
                        print('{}.{}'.format(str(idx), pred_vps))
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'gt_seg')), lane_label * 255)
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'lane')), seg_map)
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'headmap')), heatmap)
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'headmap_add')), heatmap_add)
                        vp_map = cv2.cvtColor(vp_map, cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(os.path.join(vis_path, '{}_{}.png'.format(idx, 'stack')), np.vstack((img, vp_map*255, seg_map, heatmap, heatmap_add)))

    #                     cv2.imshow('pred', img)
    #                     cv2.waitKey(0)
#         try:
#             print('loss_avg', sum(loss_avg) / len(loss_avg))
#         except:
#             pass
#         if save_predictions:
#             with open('predictions.pkl', 'wb') as handle:
#                 pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)
    def fps(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch), strict=False)
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader(self.mode)
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        vis_path = os.path.join('experiments', self.exp.name, 'vis_{}'.format(self.view))
        os.makedirs(vis_path, exist_ok=True)
        loss_avg = []
        loss_parameters = self.cfg.get_loss_parameters()
        
        fps = []
        with torch.no_grad():
            for i in range(3):
                past = []
                for idx, (images, labels, rpn_proposals, vp_idx, vp_labels, lane_labels) in enumerate(tqdm(dataloader)):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    vp_labels = vp_labels.to(self.device) 
                    lane_labels = lane_labels.to(self.device)
                    vp_idxs = []
                    for vp_label in vp_labels:
                        vp_x_y = torch.mean(torch.nonzero(vp_label, as_tuple=False).float(), 0)
                        vp_idxs.append(vp_x_y)
                    vp_idxs = torch.stack(vp_idxs)[:, [1, 0]]
                    #预测出没有消失点的分割图，消失点设为0
                    vp_idxs = torch.where(torch.isnan(vp_idxs), torch.tensor(-1e6, device = self.device), vp_idxs)
                    begin_time = time()
                    outputs, vp_logits, pred_vps,lane_logits, h_para = model(images, **test_parameters)
                    past.append(time() - begin_time)
                    prediction = model.decode(outputs, as_lanes=True)
                    predictions.extend(prediction)
                fps.append(1.0 / sum(past) * len(past))
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0], gt_vp= vp_idxs[0].cpu().numpy(), pred_vp = pred_vps[0].cpu().numpy())
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                        continue

                        cv2.imwrite(os.path.join(vis_path, str(idx)+ '.jpg'), img)
            #                     cv2.imshow('pred', img)
            #                     cv2.waitKey(0)
        print('fps', fps)
        print(max(fps))
        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)
        
    def get_train_dataloader(self, split='train'):
        print('self.mode', split)
        train_dataset = self.cfg.get_dataset(split)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=1,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self, split='test'):
        print('get_test_loader', split)
        test_dataset = self.cfg.get_dataset(split)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
