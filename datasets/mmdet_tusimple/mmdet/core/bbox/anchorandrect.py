import numpy as np
import cv2
import matplotlib.pyplot as plt 


def anchor2rect_mask_np(anchors, H=590, W=1640):
    """
    anchors : np.array[ ([x, y, w, theta]) ], theta为与x轴正方向夹角
    return ： [(x,y, w, h, angle)] 对应opencv的矩形
    
    """
    anchors = anchors.astype(np.float)
    k1 = (H-1-anchors[:, 1]) / (-anchors[:, 0] + 1e-6)
    k2 = (H-1-anchors[:, 1]) / (W-1-anchors[:, 0] + 1e-6)
    k = np.tan(anchors[:, 3] * np.pi / 180) 
    mask1 = ((anchors[:, 3] >= 90) & (k >= k1)).reshape((-1, 1))
    mask2 = ((anchors[:, 3] >= 90) & (k < k1)).reshape((-1, 1))
    mask3 = ((anchors[:, 3] <  90) & (k >k2)) .reshape((-1, 1))
    mask4 = ((anchors[:, 3] <  90) & (k <=k2)).reshape((-1, 1))
    #print(mask1, mask2, mask3, mask4)
    #左边交点的y
    p_l = np.zeros_like(anchors[:, :2])
    p_d = np.zeros_like(p_l)
    p_r = np.zeros_like(p_l)
    p_l[:, 1] = -k*anchors[:, 0] + anchors[:, 1]
    #下边交点的x
    p_d[:, 1].fill(H-1)
    p_d[:, 0] = (H-1-anchors[:, 1]) / (k + 1e-6) + anchors[:, 0]
    #右边交点的y
    p_r[:, 0].fill(W-1)
    p_r[:, 1] = k*(W-1-anchors[:, 0]) + anchors[ :, 1]
    inter_p = mask1 * p_l + (mask2 | mask3) * p_d + mask4 * p_r
    center_p = (inter_p + anchors[:, :2]) / 2
    h = np.linalg.norm(inter_p - anchors[:, :2], ord=2, axis = 1)
    w_h = (mask1 | mask2) * np.stack((h, anchors[:, 2]), axis=1)  + \
                                        ~(mask1 | mask2) * np.stack((anchors[:, 2], h), axis=1)
    angle = -anchors[:, 3] + (anchors[:, 3]>=90) * 90
    #print(torch.isnan(torch.cat((center_p, w_h, angle.unsqueeze(-1).type_as(h)), dim=1)).sum())
    
    return np.concatenate((center_p, w_h, angle[:,None]), axis = 1)

def draw_rot_rect(rect, img, trans = True):
    if trans:
        rect = ((rect[1], rect[0]), (rect[2], rect[3]), rect[4])
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)
    #draw contour里 x坐标在y前, 两列交换
    box_points = box_points[:, ::-1]
    cv2.drawContours(img, [box_points], 0, color_list[0], 2)

def plt_show(img):
    plt.figure(figsize=(40, 40))
    im2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(im2)
color_list = [(0, 0, 255), (0, 255, 0),  (0, 255, 255), (255, 0, 255),]       
        