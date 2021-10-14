import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
import random
import traceback
import tqdm
import traceback
import torch

color_list = [(0, 0, 255), (0, 255, 0),  (0, 255, 255), (255, 0, 255),]
img_dir = 'datasets/culane/'
seg_label_dir = 'datasets/culane/laneseg_label_w16_new/'
#seg_line_label_dir ='dataset/seg_line_label'
rpn_label_dir = 'datasets/culane/rpn_label_new'
vp_label_dir = 'datasets/culane/vp_label_new_32'
H = 590
W = 1640
def plt_show(img):
    plt.figure(figsize=(40, 40))
    im2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(im2)
    
def get_anno(frame_mp4_number, seg_width = 16, show = False, path2write = None, offset = 10):
    """
    display : 0, nothin; 1, show; 2, write
    return : (y1, y2, x1, w1, \theta)
    """
    #得到frame/mp4/number
    seg_img_file = os.path.join(seg_label_dir, frame_mp4_number+'.png')
    seg_img = cv2.imread(seg_img_file)
    seg_img = seg_img * 50
    seg_img_ori = seg_img.copy()
    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(seg_img_gray))
    #图中没有车道线
    if len(seg_labels) == 1:
        if(show):
            plt_show(seg_img)
        return (None, None, None)
        
    #保存直线参数的list,一条直线用3个参数表示[A，B, C]
    lines = []
    #得到覆盖标注的最小矩形
    w_annos = []
    for seg_label in seg_labels[1:]:
        rect, box_points = get_min_rect(seg_img_gray, seg_label)
        y_center = rect[0][0];
        x_center = rect[0][1];
        w=rect[1][0]
        h=rect[1][1]
##      
##
## 
        w_annos.append(min(w, h))
        angle = rect[2]
        #求这个矩形代表的直线
        angle_y = angle_convert(w, h, angle)
        angle_x = angle_convert_x(w, h, angle)
        A, B, C = get_line_equation(y_center, x_center, angle_y)
        lines.append([A, B, C, angle_x])
        
    #只有一条车道线的情况
    if(len(lines) == 1):
        x_anno = x_center - max(w, h)/2 * np.cos(angle_x*np.pi/180)
        y_l = y_r = y_center - max(w, h)/2 * np.sin(angle_x*np.pi/180)
        if show:
            cv2.circle(seg_img, (int(x_anno), int(y_l)), 3, (0, 255, 0), thickness=3)
            plt_show(seg_img)
        return (y_l, y_r, [[x_anno,  w_annos[0], angle_x]])
        
        
        
    #求交点
    x_inter_list=[]
    y_inter_list=[]
    for i in range(len(lines)-1):
        for j in range(i+1, len(lines)):
            #print(lines[i], lines[j])
            x_inter, y_inter = get_inter_point(lines[i][0], lines[i][1], lines[i][2], lines[j][0], lines[j][1], lines[j][2])
            x_inter_list.append(x_inter)
            y_inter_list.append(y_inter)

    #一个点直接回归直线
    if(len(x_inter_list) == 1):
        seg_img_modify_0 = seg_img.copy()
        y_l = y_inter_list[0]
        y_r = y_inter_list[0]
#       cv2.line(seg_img_modify_0, (0, int(y_l)), (seg_img.shape[1], int(y_r)), (255, 0, 0), 3)
        y_l = y_inter_list[0]
        y_r = y_inter_list[0]
        if show:
            cv2.line(seg_img, (0, y_l), (seg_img.shape[1], y_r),  (0, 0 , 255), 3 )
            cv2.circle(seg_img,  (x_inter_list[0], y_l), 3, (0, 255, 0), thickness=3 )
            plt_show(seg_img)
        return ( y_l, y_r, [[x_inter_list[0], w_annos[0], 0]] )
       
    #得到 y=kx+b 的[k, b]
    #kx-y+b=0
    else:
        line_anno_list = []
        x_annos = []
        seg_img_modify_10 = seg_img.copy()
        horizon_line ,y_l, y_r = modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify_10, offset=offset)
        for line, w_anno in zip(lines, w_annos):
            #print(line, horizon_line)
            x_anno, _ = get_inter_point(line[0], line[1], line[2], horizon_line[0], -1, horizon_line[1])
            x_annos.append(x_anno)
            # x ,w ,theta
            line_anno_list.append([  x_anno  , w_anno, line[3]])
            
        if show:
            cv2.line(seg_img, (0, int(y_l)), (seg_img.shape[1], int(y_r)), (0, 0 , 255), 3 )
            for line, x_anno in zip(lines, x_annos):
                #矩形代表直线在水平分割线上交点
                y_anno = (-line[2] + -line[0] * x_anno)/line[1]
                cv2.circle(seg_img,  (int(x_anno), int(y_anno)), 3, (0, 255, 0), thickness=3 )
                plt_show(seg_img)
        
        return (y_l, y_r, line_anno_list)




def plt_show(img):
    im2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(im2)
    

def show_bounding_boxes(seg_anno_file):
    img = cv2.imread(seg_anno_file)
    img = img * 50
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(img_gray))
    #得到覆盖标注的最小矩形
    for seg_label in seg_labels[1:]:
        x,y,w,h =get_boundingRect(img_gray, seg_label)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
        _, box_points = get_min_rect(img_gray, seg_label)
        cv2.drawContours(img, [box_points], 0, (0, 0, 255), 2)
    plt_show(img)


def get_boundingRect(img_gray, seg_label):
    #画最小标准形
    cnt = (img_gray==seg_label).astype(np.uint8)
    #cnt 二维图像或点集
    x, y, w, h = cv2.boundingRect(cnt)
    return x,y,w,h




# cv2.minAreaRect返回的width，height，angle比较难理解
# https://stackoverflow.com/questions/24073127/opencvs-rotatedrect-angle-does-not-provide-enough-information
# https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned/21427814#21427814
# 得到由y轴逆时针转到直线的角度
def angle_convert(width, height, angle):
    assert width!=height

    return 180+angle if width>height  else angle + 90
    return angle


# 得到由x轴正方向夹角
def angle_convert_x(width, height, angle):
    assert width!=height
    return 90-angle if width>height  else -angle
    return angle

# 返回在cv2的坐标下与y轴的夹角, 由点斜式
# angle是由y轴转到x轴的方向转到直线的角度
def get_line_equation(y0, x0, angle):
    if angle == 90:
        return (0, 1, -y0);
    else:
        k=np.tan(angle*np.pi/180)
    #两点式
    #x-x0=k(y-y0)
    #x-ky+ky0-x0=0
    return (1, -k, k*y0-x0)


def get_line_equation_x(x0, y0, angle):
    if angle == 90:
        return (0, 1, -y0);
    else:
        k=np.tan(angle*np.pi/180)
    #两点式
    #x-x0=k(y-y0)
    #x-ky+ky0-x0=0
    return (k, 1, -k*x0-y0)

def draw_line_equation_ori(img, A, B, C,  y_max, color=(0, 255, 0), x_max=None):
    #首先处理垂直y轴, A=0时需要x_max
    if A == 0:
        assert x_max
        for x in range(x_max):
            cv2.circle(img, (x, -int(C)), 5, color, 4)
        return
    for y in range(y_max):
        x = (-B*y - C)/A
        cv2.circle(img, (int(x), y), 1, color, 4)

# TODO : 考虑直线与矩形的相交具体怎么样， cv2.line 能画两个点在图外吗
def draw_line_equation(img, A, B, C, y_max, color = (0, 255, 0), x_max = None):
#def draw_line_equation(img, A, B, C, y_max = H, color = (0, 255, 0), x_max = None):
    assert (A!=0 or B!=0)
    if A==0:
        y_lr = -C/B
        cv2.line(img, (0, y_lr), (x_max, y_lr), color, 3)
    else:
        x1 = (-B*0 - C)/A
        x2 = (-B*y_max - C)/A
        cv2.line(img, (int(x1), 0), (int(x2), y_max), color, 3)

def get_inter_point(A1, B1, C1, A2, B2, C2):
#基于一般式，一般式 A,B不全为0
    assert (A1!=0 or B1!=0) and (A2!=0 or B2!=0)
    m=A1*B2-A2*B1
    if m==0:
        return (-1000, -1000)
    else:
        x=(C2*B1-C1*B2)/m
        y=(C1*A2-C2*A1)/m
        return x,y

#在点集的两边增加点集最小x，最大x的两个在一条水平线上的点
def modify_points(x_inter_list, y_inter_list, seg_img, draw = False, offset=0):
    y_inter_center = np.mean(y_inter_list)
    x_inter_min = np.min(x_inter_list) - offset
    x_inter_max = np.max(x_inter_list) + offset
    horizon_line = np.polyfit(x_inter_list, y_inter_list, deg=1)
    x_inter_list.append(x_inter_min)
    y_inter_list.append(y_inter_center)
    x_inter_list.append(x_inter_max)
    y_inter_list.append(y_inter_center)
    
    
    horizon_line = np.polyfit(x_inter_list, y_inter_list, deg=1)
    y_l = horizon_line[1]
    y_r = horizon_line[0] * seg_img.shape[1] + horizon_line[1]
    if draw:
        cv2.circle(seg_img, (int(x_inter_min),int(y_inter_center)), 3, (0, 255, 255), thickness=3)
        cv2.circle(seg_img, (int(x_inter_max),int(y_inter_center)), 3, (0, 255, 255), thickness=3)
        cv2.line(seg_img, (0, int(y_l)), (seg_img.shape[1], int(y_r)), (255, 0, 0), 3)
    return horizon_line , y_l, y_r


#对比不同方法得到的horizon_line
def contrast_horizon_line(frame_mp4_number, display, path2write = None):
    """
    display : 0, nothin; 1, show; 2, write
    """
    ori_img  = cv2.imread(os.path.join(img_dir, frame_mp4_number+'.jpg'))
    seg_anno_file = os.path.join(seg_label_dir, frame_mp4_number+'.png')
    seg_img = cv2.imread(seg_anno_file)
    seg_img = seg_img * 50
    seg_img_ori = seg_img.copy()
    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(seg_img_gray))
    if len(seg_labels) == 1:
        return -1,-1
    #保存直线参数的list,一条直线用3个参数表示[A，B, C]
    lines = []
    #得到覆盖标注的最小矩形
    for seg_label in seg_labels[1:]:
        x,y,w,h =get_boundingRect(seg_img_gray, seg_label)
#         if display>0:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        rect, box_points = get_min_rect(seg_img_gray, seg_label)
        y_center = rect[0][0];
        x_center = rect[0][1];
        w=rect[1][0]
        h=rect[1][1]
        angle = rect[2]
        if display>0:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 1)
        #求这个矩形代表的直线
        angle = angle_convert(w, h, angle)
        A, B, C = get_line_equation(y_center, x_center, angle)
        if display>0:
             draw_line_equation(seg_img, A, B ,C, seg_img.shape[0])
        lines.append([A, B, C])
    #求交点
    x_inter_list=[]
    y_inter_list=[]
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            x_inter, y_inter = get_inter_point(lines[i][0], lines[i][1], lines[i][2], lines[j][0], lines[j][1], lines[j][2])
            x_inter_list.append(x_inter)
            y_inter_list.append(y_inter)
            if display>0:
                cv2.circle(seg_img, (int(x_inter),int(y_inter)), 3, (0, 255, 0), thickness=3)
   

    #改进前
    seg_img_copy = seg_img.copy()
    horizon_line = np.polyfit(x_inter_list, y_inter_list, deg=1)
    y_l = horizon_line[1]
    y_r = horizon_line[0] * seg_img.shape[1] + horizon_line[1]
    cv2.line(seg_img_copy, (0, int(y_l)), (seg_img.shape[1], int(y_r)), (255, 0, 0), 3)
    #改进前
    
    
    
    #一个点直接回归直线
    if(len(x_inter_list) == 1):
        seg_img_modify_0 = seg_img.copy()
        y_l = y_inter_list[0]
        y_r = y_inter_list[0]
        cv2.line(seg_img_modify_0, (0, int(y_l)), (seg_img.shape[1], int(y_r)), (255, 0, 0), 3)
        y_l = y_inter_list[0]
        y_r = y_inter_list[0]
        if display==1:
            plt_show(np.vstack((  np.hstack((seg_img_ori, seg_img_copy)),  np.hstack((ori_img, seg_img_modify_0))  )) )

        elif display == 2:
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            cv2.imwrite(os.path.join(path2write, '_'.join(name_split_list[-3:])), \
                        np.vstack((  np.hstack((seg_img_ori, seg_img_copy)),  np.hstack((ori_img, seg_img_modify_0))  )) )
    #得到 y=kx+b 的[k, b]
    #kx-y+b=0
    else:
        #添加点集左右两边的点
        seg_img_modify_0 = seg_img.copy()
        modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify_0 )
        #offset 5
        seg_img_modify_5 = seg_img.copy()
        modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify_5, 5)
        #offset 10
        seg_img_modify_10 = seg_img.copy()
        horizon_line ,y_l, y_r = modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify_10, 20)

        if display==1:
            plt_show(np.vstack((np.hstack((seg_img_ori, seg_img_copy, seg_img_modify_0)),  \
                                np.hstack((ori_img, seg_img_modify_5, seg_img_modify_10)))))

        elif display == 2:
            if not os.path.exists(path2write):
                os.makedirs(path2write)
            cv2.imwrite(os.path.join(path2write, '_'.join(name_split_list[-3:])), \
                        np.vstack((np.hstack((seg_img_ori, seg_img_copy, seg_img_modify_0)),  \
                                   np.hstack((ori_img, seg_img_modify_5, seg_img_modify_10)))))
    return y_l, y_r


# y1, y2
# x1, w1, \theta




def get_anno(frame_mp4_number, seg_width = 16, show = False, path2write = None, offset = 10):
    """
    display : 0, nothin; 1, show; 2, write
    return : (y1, y2, x1, w1, \theta)
    """
    #得到frame/mp4/number
    seg_img_file = os.path.join(seg_label_dir, frame_mp4_number+'.png')
    seg_img = cv2.imread(seg_img_file)
    H, W = seg_img.shape[:2]
    seg_img = seg_img * 50
    seg_img_ori = seg_img.copy()
    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(seg_img_gray))
    #图中没有车道线
    if len(seg_labels) == 1:
        if(show):
            plt_show(seg_img)
        return (None, None, None)
        
    #保存直线参数的list,一条直线用3个参数表示[A，B, C]
    lines = []
    #得到覆盖标注的最小矩形
    w_annos = []
    for seg_label in seg_labels[1:]:
        rect, box_points, (x3, x4) = get_min_rect(seg_img_gray, seg_label)
        cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
        
        y_center = rect[0][0];
        x_center = rect[0][1];
        w=rect[1][0]
        h=rect[1][1]
        angle = rect[2]
        if display>0:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 1)
        #求这个矩形代表的直线
        angle = angle_convert(w, h, angle)
        A, B, C = get_line_equation(y_center, x_center, angle)
        if display>0:
             draw_line_equation(seg_img, A, B ,C, seg_img.shape[0])
        lines.append([A, B, C])
        A, B, C = get_line_equation(y_center, x_center, angle_y)
        lines.append([A, B, C, angle_x])
    plt_show(seg_img)
    #只有一条车道线的情况
    if(len(lines) == 1):
        x_anno = x_center - max(w, h)/2 * np.cos(angle_x*np.pi/180)
        y_l = y_r = y_center - max(w, h)/2 * np.sin(angle_x*np.pi/180)
        if show:
            cv2.circle(seg_img, (int(x_anno), int(y_l)), 3, (0, 255, 0), thickness=3)
            plt_show(seg_img)
        return (y_l, y_r, [[x_anno,  w_annos[0], angle_x]])
        
        
        
    #求交点
    x_inter_list=[]
    y_inter_list=[]
    for i in range(len(lines)-1):
        for j in range(i+1, len(lines)):
            #print(lines[i], lines[j])
            x_inter, y_inter = get_inter_point(lines[i][0], lines[i][1], lines[i][2], lines[j][0], lines[j][1], lines[j][2])
            x_inter_list.append(x_inter)
            y_inter_list.append(y_inter)

    #一个点直接回归直线
    if(len(x_inter_list) == 1):
        seg_img_modify_0 = seg_img.copy()
        y_l = y_inter_list[0]
        y_r = y_inter_list[0]
#       cv2.line(seg_img_modify_0, (0, int(y_l)), (seg_img.shape[1], int(y_r)), (255, 0, 0), 3)
        y_l = y_inter_list[0]
        y_r = y_inter_list[0]
        if show:
            cv2.line(seg_img, (0, y_l), (seg_img.shape[1], y_r),  (0, 0 , 255), 3 )
            cv2.circle(seg_img,  (x_inter_list[0], y_l), 3, (0, 255, 0), thickness=3 )
            plt_show(seg_img)
        return ( y_l, y_r, [[x_inter_list[0], w_annos[0], 0]] )
       
    #得到 y=kx+b 的[k, b]
    #kx-y+b=0
    else:
        line_anno_list = []
        x_annos = []
        seg_img_modify_10 = seg_img.copy()
        horizon_line ,y_l, y_r = modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify_10, offset=offset)
        for line, w_anno in zip(lines, w_annos):
            #print(line, horizon_line)
            x_anno, _ = get_inter_point(line[0], line[1], line[2], horizon_line[0], -1, horizon_line[1])
            x_annos.append(x_anno)
            # x ,w ,theta
            line_anno_list.append([  x_anno  , w_anno, line[3]])
            
        if show:
            cv2.line(seg_img, (0, int(y_l)), (seg_img.shape[1], int(y_r)), (0, 0 , 255), 3 )
            for line, x_anno in zip(lines, x_annos):
                #矩形代表直线在水平分割线上交点
                y_anno = (-line[2] + -line[0] * x_anno)/line[1]
                cv2.circle(seg_img, (int(x_anno), int(y_anno)), 3, (0, 255, 0), thickness=3 )
                plt_show(seg_img)
        
        return (y_l, y_r, line_anno_list)
def distance(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def draw_rot_rect(rect, img):
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)
    #draw contour里 x坐标在y前, 两列交换
    box_points = box_points[:, ::-1]
    cv2.drawContours(img, [box_points], 0, color_list[0], 2)

#求在图内的最小矩形
def get_min_rect(img_gray, seg_label, seg_img):
    points = np.where(img_gray==seg_label)
    points = np.stack(points, axis=1)
    #输入点集
    rect = cv2.minAreaRect(points)
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)
    #draw contour里 x坐标在y前, 两列交换
    box_points = box_points[:, ::-1]
    
    y_center = rect[0][0];
    x_center = rect[0][1];
    w=rect[1][0]
    h=rect[1][1]
    w_anno = min(w, h)
    h_anno = max(w, h)
    angle = rect[2]
    #求这个矩形代表的直线
    angle_y = angle_convert(w, h, angle)
    angle_x = angle_convert_x(w, h, angle)
    x_upper_center = x_center - h_anno/2 * np.cos(angle_x * np.pi / 180)
    y_upper_center = y_center - h_anno/2 * np.sin(angle_x * np.pi / 180)
    #短边与x轴的夹角
    angle_v_x = angle_x + 90 if angle_x < 90 else angle_x - 90
    x1 = x_upper_center + w_anno/2 * np.cos(angle_v_x * np.pi / 180)
    x2 = x_upper_center - w_anno/2 * np.cos(angle_v_x * np.pi / 180)
    y1 = y_upper_center + w_anno/2 * np.sin(angle_v_x * np.pi / 180)
    y2 = y_upper_center - w_anno/2 * np.sin(angle_v_x * np.pi / 180)
    #check短边两点是否正确
#     cv2.circle(seg_img, (int(x_upper_center),int(y_upper_center)), 1, (0, 255, 0), thickness=2)
#     cv2.circle(seg_img, (int(x1),int(y1)), 1, (0, 255, 0), thickness=2)
#     cv2.circle(seg_img, (int(x2),int(y2)), 1, (0, 255, 0), thickness=2)
    line_1 = get_line_equation(y1, x1, angle_y)
    line_2 = get_line_equation(y2, x2, angle_y)
    #draw_line_equation(img, A, B, C, y_max = H, color = (0, 255, 0), x_max = None):
    #def get_inter_point(A1, B1, C1, A2, B2, C2):
#     draw_line_equation(seg_img, line_1[0], line_1[1], line_1[2], x_max = W)
#     draw_line_equation(seg_img, line_2[0], line_2[1], line_2[2], x_max = W)

    #与y=H-1交点
    x_inter_1_h, y_inter_1_h = get_inter_point(line_1[0], line_1[1], line_1[2], 0, 1, -H+1)
    x_inter_2_h, y_inter_2_h = get_inter_point(line_2[0], line_2[1], line_2[2], 0, 1, -H+1)
    
    #与x=0或W-1交点
    W_or_0 = 0 if w>h else W-1
    x_inter_1_w, y_inter_1_w = get_inter_point(line_1[0], line_1[1], line_1[2], 1, 0, -W_or_0)
    x_inter_2_w, y_inter_2_w = get_inter_point(line_2[0], line_2[1], line_2[2], 1, 0, -W_or_0)
#         cv2.circle(seg_img, (int(x_inter_1),int(y_inter_1)), 1, (0, 255, 0), thickness=2)
#         cv2.circle(seg_img, (int(x_inter_2),int(y_inter_2)), 1, (0, 255, 0), thickness=2)
    if distance([x1,y1], [x_inter_1_h, y_inter_1_h]) < distance([x1,y1], [x_inter_1_w, y_inter_1_w]):
        x_inter_1, y_inter_1 = x_inter_1_h, y_inter_1_h
        distance_1 = distance([x1,y1], [x_inter_1_h, y_inter_1_h])
    else:
        x_inter_1, y_inter_1 = x_inter_1_w, y_inter_1_w
        distance_1 = distance([x1,y1], [x_inter_1_w, y_inter_1_w])

    if distance([x2,y2], [x_inter_2_h, y_inter_2_h]) < distance([x2,y2], [x_inter_2_w, y_inter_2_w]):
        x_inter_2, y_inter_2 = x_inter_2_h, y_inter_2_h
        distance_2 = distance([x2, y2], [x_inter_2_h, y_inter_2_h])
    else:
        x_inter_2, y_inter_2 = x_inter_2_w, y_inter_2_w
        distance_2 = distance([x2,y2], [x_inter_2_w, y_inter_2_w])

    if distance_1 < distance_2 : 
        cut_h = h_anno - distance_1
        x3, y3 = x_inter_1, y_inter_1
        x4 = x3 - w_anno * np.cos(angle_v_x * np.pi / 180)
        y4 = y3 - w_anno * np.sin(angle_v_x * np.pi / 180)
    else:
        cut_h = h_anno - distance_2
        x4, y4 = x_inter_2, y_inter_2
        x3 = x4 + w_anno * np.cos(angle_v_x * np.pi / 180)
        y3 = y4 + w_anno * np.sin(angle_v_x * np.pi / 180)
    #cv2.circle(seg_img, (int(x3),int(y3)), 3, (0, 255, 0), thickness=2) 
    #cv2.circle(seg_img, (int(x4),int(y4)), 3, (0, 0, 255), thickness=2) 
#     print('x3, y3', x3, y3)
#     print('x4, y4', x4, y4)

    if min(distance_1, distance_2) <= h_anno:
        new_x_center = x_center - cut_h / 2 * np.cos(angle_x * np.pi / 180)
        new_y_center = y_center - cut_h / 2 * np.sin(angle_x * np.pi / 180)
        if w>h:
            new_rect = ((new_y_center, new_x_center), (w-cut_h, h), angle)
        else:
            new_rect = ((new_y_center, new_x_center), (w, h-cut_h), angle)
        new_box_points = cv2.boxPoints(new_rect)
        new_box_points = np.int0(new_box_points)
        #draw contour里 x坐标在y前, 两列交换
        new_box_points = new_box_points[:, ::-1]
        return new_rect, new_box_points
    else:
        return rect, box_points


#求在图内的最小矩形, 求与矩形的交点连线
def get_min_rect_v2(img_gray, seg_label, seg_img):
    points = np.where(img_gray==seg_label)
    points = np.stack(points, axis=1)
    #输入点集
    rect = cv2.minAreaRect(points)
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)
    #draw contour里 x坐标在y前, 两列交换
    box_points = box_points[:, ::-1]
    #cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 1)
    #plt_show(seg_img)
    y_center = rect[0][0];
    x_center = rect[0][1];
    w=rect[1][0]
    h=rect[1][1]
    w_anno = min(w, h)
    h_anno = max(w, h)
    angle = rect[2]
    #求这个矩形代表的直线
    angle_y = angle_convert(w, h, angle)
    angle_x = angle_convert_x(w, h, angle)
    x_upper_center = x_center - h_anno/2 * np.cos(angle_x * np.pi / 180)
    y_upper_center = y_center - h_anno/2 * np.sin(angle_x * np.pi / 180)
    #长边中心线
    line = get_line_equation(y_center, x_center, angle_y)
    #draw_line_equation(seg_img, line[0], line[1], line[2], W)
     #与y=H-1交点
    x_inter_h, y_inter_h = get_inter_point(line[0], line[1], line[2], 0, 1, -H+1)
    #与x=0或W-1交点
    W_or_0 = 0 if w>h else W-1
    x_inter_w, y_inter_w = get_inter_point(line[0], line[1], line[2], 1, 0, -W_or_0)
    if distance([x_center,y_center], [x_inter_h, y_inter_h]) < distance([x_center, y_center], [x_inter_w, y_inter_w]):
        x_inter, y_inter = x_inter_h, y_inter_h
        short_d = distance([x_center, y_center], [x_inter_h, y_inter_h])
    else:
        x_inter, y_inter = x_inter_w, y_inter_w
        short_d = distance([x_center, y_center], [x_inter_w, y_inter_w])
    new_x_center = (x_upper_center + x_inter) /2 
    new_y_center = (y_upper_center + y_inter) /2 
    if w>h:
        w = short_d + w/2
    else:
        h = short_d + h/2
    new_rect = ((new_y_center, new_x_center), (w, h), angle)
    new_box_points = cv2.boxPoints(new_rect)
    new_box_points = np.int0(new_box_points)
    #draw contour里 x坐标在y前, 两列交换
    new_box_points = new_box_points[:, ::-1]
    return new_rect, new_box_points  
    
    

#根据矩形的长边变化的宽度，变化旋转矩形
def modify_rect(rect,  x_inter, y_inter):
    
    x_center = rect[0][1]
    y_center = rect[0][0]
    w=rect[1][0]
    h=rect[1][1]
    angle = rect[2]
    add_h = distance([x_center, y_center], [x_inter, y_inter]) - max(w, h)/2
    angle_x = angle_convert_x(w, h, angle)
    new_x_center = x_center - add_h / 2 * np.cos(angle_x * np.pi / 180)
    new_y_center = y_center - add_h / 2 * np.sin(angle_x * np.pi / 180)
    if w>h:
        w=w+add_h
    else:
        h=h+add_h
    new_rect = ((new_y_center , new_x_center), (w, h), rect[2])
    return new_rect

def rect2anchor(rect):
    """
    rect : opencv 形式
    anchor : [x_u, y_u, w, h, angle_x] w指较短的一边
    """
    y_center = rect[0][0];
    x_center = rect[0][1];
    w=rect[1][0]
    h=rect[1][1]
    w_anno = min(w, h)
    h_anno = max(w, h)
    angle = rect[2]
    #求这个矩形代表的直线
    angle_x = angle_convert_x(w, h, angle)
    x_upper_center = x_center - h_anno/2 * np.cos(angle_x * np.pi / 180)
    y_upper_center = y_center - h_anno/2 * np.sin(angle_x * np.pi / 180)
    return [x_upper_center, y_upper_center, w_anno, h_anno, angle_x]

def get_anno_rpn(frame_mp4_number, seg_width = 16, show = False, path2write = None, offset = 10):
    """
    display : 0, nothin; 1, show; 2, write
    return : (y1, y2, x1, w1, \theta)
    """
    #得到frame/mp4/number
    seg_img_file = os.path.join(seg_label_dir, frame_mp4_number+'.png')
    seg_img = cv2.imread(seg_img_file)
    H, W = seg_img.shape[:2]
    seg_img = seg_img * 50
    seg_img_ori = seg_img.copy()
    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(seg_img_gray))
    print(seg_labels)
    #图中没有车道线
    if len(seg_labels) == 1:
        if(show):
            plt_show(seg_img)
        return None
        
    #保存直线参数的list,一条直线用3个参数表示[A，B, C]
    lines = []
    rects = []
    #得到覆盖标注的最小矩形
    for seg_label in seg_labels[1:]:
        print(seg_label)
        #rect, box_points = get_min_rect_ori(seg_img_gray, seg_label)
        #cv2.drawContours(seg_img, [box_points], 0, (0, 255, 0), 2)
        try:
            rect, box_points = get_min_rect_v2(seg_img_gray, seg_label, seg_img)
        except:
            continue
        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
        rects.append(rect)
        y_center = rect[0][0];
        x_center = rect[0][1];
        w=rect[1][0]
        h=rect[1][1]
#         w_anno = min(w, h)
#         h_anno = max(w, h)
        angle = rect[2]
        #求这个矩形代表的直线
        angle_y = angle_convert(w, h, angle)
        angle_x = angle_convert_x(w, h, angle)
        A, B, C = get_line_equation(y_center, x_center, angle_y)
        lines.append([A, B, C, angle_x])
    #plt_show(seg_img)
    #只有一条车道线的情况
    if(len(lines) == 1):
        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
            plt_show(seg_img)
        return rects
        
        
        
    #求交点
    x_inter_list=[]
    y_inter_list=[]
    for i in range(len(lines)-1):
        for j in range(i+1, len(lines)):
            #print(lines[i], lines[j])
            x_inter, y_inter = get_inter_point(lines[i][0], lines[i][1], lines[i][2], lines[j][0], lines[j][1], lines[j][2])
            x_inter_list.append(x_inter)
            y_inter_list.append(y_inter)

    #一个点直接回归直线
    if(len(x_inter_list) == 1):
        seg_img_modify = seg_img.copy()
        new_rects=[]
        for rect in rects:
            new_rect = modify_rect(rect, x_inter_list[0], y_inter_list[0])
            new_rects.append(new_rect)
            if show:
                cv2.circle(seg_img_modify, (int(x_inter_list[0]),int(y_inter_list[0])), 3, (0, 255, 0), thickness=2)
                new_box_points = cv2.boxPoints(new_rect)
                new_box_points = np.int0(new_box_points)
                #draw contour里 x坐标在y前, 两列交换
                new_box_points = new_box_points[:, ::-1]
                cv2.drawContours(seg_img_modify, [new_box_points], 0, (0, 0, 255), 2)
                plt_show(seg_img_modify)
        return new_rects
            
       
    #得到 y=kx+b 的[k, b]
    #kx-y+b=0
    else:
        new_rects = []
        line_anno_list = []
        x_annos = []
        seg_img_modify = seg_img.copy()
        horizon_line ,y_l, y_r = modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify, offset=offset)
        for i, (line, rect) in enumerate(zip(lines, rects)):
            #print(line, horizon_line)
            x_anno, y_anno = get_inter_point(line[0], line[1], line[2], horizon_line[0], -1, horizon_line[1])
            if show:
                cv2.circle(seg_img_modify, (int(x_anno),int(y_anno)), 3, (0, 255, 0), thickness=2) 
            new_rect = modify_rect(rect, x_anno, y_anno)
            new_rects.append(new_rect)
            if show:   
                new_box_points = cv2.boxPoints(new_rect)
                new_box_points = np.int0(new_box_points)
                #draw contour里 x坐标在y前, 两列交换
                new_box_points = new_box_points[:, ::-1]
                cv2.drawContours(seg_img_modify, [new_box_points], 0, color_list[i], 2)
            
        if show:
            for line, x_anno in zip(lines, x_annos):
                #矩形代表直线在水平分割线上交点
                y_anno = (-line[2] + -line[0] * x_anno)/line[1]
                cv2.circle(seg_img_modify,  (int(x_anno), int(y_anno)), 3, (0, 255, 0), thickness=3 )
            plt_show(seg_img_modify)
        
        return new_rects
#得到上顶点，x,y,w,h,theta(与x轴)
def get_anno_rpn_anchor(frame_mp4_number, seg_width = 16, show = False, path2write = None, offset = 10):
    """
    display : 0, nothin; 1, show; 2, write
    return : (y1, y2, x1, w1, \theta)
    """
    #得到frame/mp4/number
    seg_img_file = os.path.join(seg_label_dir, frame_mp4_number+'.png')
    seg_img = cv2.imread(seg_img_file)
    H, W = seg_img.shape[:2]
    seg_img = seg_img * 50
    seg_img_ori = seg_img.copy()
    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(seg_img_gray))
    #图中没有车道线
    if len(seg_labels) == 1:
        if(show):
            plt_show(seg_img)
        return None
        
    #保存直线参数的list,一条直线用3个参数表示[A，B, C]
    lines = []
    rects = []
    #得到覆盖标注的最小矩形
    for seg_label in seg_labels[1:]:
        #rect, box_points = get_min_rect_ori(seg_img_gray, seg_label)
        #cv2.drawContours(seg_img, [box_points], 0, (0, 255, 0), 2)
        try:
            rect, box_points = get_min_rect_v2(seg_img_gray, seg_label, seg_img)
        except:
            print(frame_mp4_number)
            continue
        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
        rects.append(rect)
        y_center = rect[0][0];
        x_center = rect[0][1];
        w=rect[1][0]
        h=rect[1][1]
#         w_anno = min(w, h)
#         h_anno = max(w, h)
        angle = rect[2]
        #求这个矩形代表的直线
        angle_y = angle_convert(w, h, angle)
        angle_x = angle_convert_x(w, h, angle)
        A, B, C = get_line_equation(y_center, x_center, angle_y)
        lines.append([A, B, C, angle_x])
    #plt_show(seg_img)
    #只有一条车道线的情况
    if(len(lines) == 1):
        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
            plt_show(seg_img)
        anchor = rect2anchor(rects[0])
        return [anchor[1], anchor[1], [anchor]]
        
        
        
    #求交点
    x_inter_list=[]
    y_inter_list=[]
    for i in range(len(lines)-1):
        for j in range(i+1, len(lines)):
            #print(lines[i], lines[j])
            x_inter, y_inter = get_inter_point(lines[i][0], lines[i][1], lines[i][2], lines[j][0], lines[j][1], lines[j][2])
            x_inter_list.append(x_inter)
            y_inter_list.append(y_inter)

    #一个点直接回归直线
    if(len(x_inter_list) == 1):
        seg_img_modify = seg_img.copy()
        new_rects=[]
        for rect in rects:
            new_rect = modify_rect(rect, x_inter_list[0], y_inter_list[0])
            new_rects.append(new_rect)
            if show:
                cv2.circle(seg_img_modify, (int(x_inter_list[0]),int(y_inter_list[0])), 3, (0, 255, 0), thickness=2)
                new_box_points = cv2.boxPoints(new_rect)
                new_box_points = np.int0(new_box_points)
                #draw contour里 x坐标在y前, 两列交换
                new_box_points = new_box_points[:, ::-1]
                cv2.drawContours(seg_img_modify, [new_box_points], 0, (0, 0, 255), 2)
                plt_show(seg_img_modify)
        anchors = [rect2anchor(rect) for rect in new_rects]
        return [y_inter_list[0],y_inter_list[0], anchors]
            
       
    #得到 y=kx+b 的[k, b]
    #kx-y+b=0
    else:
        new_rects = []
        line_anno_list = []
        x_annos = []
        seg_img_modify = seg_img.copy()
        horizon_line ,y_l, y_r = modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify, offset=offset)
        for i, (line, rect) in enumerate(zip(lines, rects)):
            #print(line, horizon_line)
            x_anno, y_anno = get_inter_point(line[0], line[1], line[2], horizon_line[0], -1, horizon_line[1])
            if show:
                cv2.circle(seg_img_modify, (int(x_anno),int(y_anno)), 3, (0, 255, 0), thickness=2) 
            new_rect = modify_rect(rect, x_anno, y_anno)
            new_rects.append(new_rect)
            if show:   
                new_box_points = cv2.boxPoints(new_rect)
                new_box_points = np.int0(new_box_points)
                #draw contour里 x坐标在y前, 两列交换
                new_box_points = new_box_points[:, ::-1]
                cv2.drawContours(seg_img_modify, [new_box_points], 0, color_list[i], 2)
            
        if show:
            for line, x_anno in zip(lines, x_annos):
                #矩形代表直线在水平分割线上交点
                y_anno = (-line[2] + -line[0] * x_anno)/line[1]
                cv2.circle(seg_img_modify,  (int(x_anno), int(y_anno)), 3, (0, 255, 0), thickness=3 )
            plt_show(seg_img_modify)
        anchors = [rect2anchor(rect) for rect in new_rects]
        return [y_l, y_r, anchors]

    
#得到上顶点，x,y,w,h,theta(与x轴)
def get_anno_rpn_anchor(frame_mp4_number, seg_width = 16, show = False, path2write = None, offset = 10):
    """
    display : 0, nothin; 1, show; 2, write
    return : (y1, y2, x1, w1, \theta)
    """
    #得到frame/mp4/number
    seg_img_file = os.path.join(seg_label_dir, frame_mp4_number+'.png')
    seg_img = cv2.imread(seg_img_file)
    H, W = seg_img.shape[:2]
    seg_img = seg_img * 50
    seg_img_ori = seg_img.copy()
    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(seg_img_gray))
    #图中没有车道线
    if len(seg_labels) == 1:
        if(show):
            plt_show(seg_img)
        return None
        
    #保存直线参数的list,一条直线用3个参数表示[A，B, C]
    lines = []
    rects = []
    #得到覆盖标注的最小矩形
    for seg_label in seg_labels[1:]:
        #rect, box_points = get_min_rect_ori(seg_img_gray, seg_label)
        #cv2.drawContours(seg_img, [box_points], 0, (0, 255, 0), 2)
        rect, box_points = get_min_rect_v2(seg_img_gray, seg_label, seg_img)

        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
        rects.append(rect)
        y_center = rect[0][0];
        x_center = rect[0][1];
        w=rect[1][0]
        h=rect[1][1]
#         w_anno = min(w, h)
#         h_anno = max(w, h)
        angle = rect[2]
        #求这个矩形代表的直线
        angle_y = angle_convert(w, h, angle)
        angle_x = angle_convert_x(w, h, angle)
        A, B, C = get_line_equation(y_center, x_center, angle_y)
        lines.append([A, B, C, angle_x])
    #plt_show(seg_img)
    #只有一条车道线的情况
    if(len(lines) == 1):
        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
            plt_show(seg_img)
        anchor = rect2anchor(rects[0])
        return [anchor[1], anchor[1], [anchor]]
        
        
        
    #求交点
    x_inter_list=[]
    y_inter_list=[]
    for i in range(len(lines)-1):
        for j in range(i+1, len(lines)):
            #print(lines[i], lines[j])
            x_inter, y_inter = get_inter_point(lines[i][0], lines[i][1], lines[i][2], lines[j][0], lines[j][1], lines[j][2])
            x_inter_list.append(x_inter)
            y_inter_list.append(y_inter)

    #一个点直接回归直线
    if(len(x_inter_list) == 1):
        seg_img_modify = seg_img.copy()
        new_rects=[]
        for rect in rects:
            new_rect = modify_rect(rect, x_inter_list[0], y_inter_list[0])
            new_rects.append(new_rect)
            if show:
                cv2.circle(seg_img_modify, (int(x_inter_list[0]),int(y_inter_list[0])), 3, (0, 255, 0), thickness=2)
                new_box_points = cv2.boxPoints(new_rect)
                new_box_points = np.int0(new_box_points)
                #draw contour里 x坐标在y前, 两列交换
                new_box_points = new_box_points[:, ::-1]
                cv2.drawContours(seg_img_modify, [new_box_points], 0, (0, 0, 255), 2)
                plt_show(seg_img_modify)
        anchors = [rect2anchor(rect) for rect in new_rects]
        return [y_inter_list[0],y_inter_list[0], anchors]
            
       
    #得到 y=kx+b 的[k, b]
    #kx-y+b=0
    else:
        new_rects = []
        line_anno_list = []
        x_annos = []
        seg_img_modify = seg_img.copy()
        horizon_line ,y_l, y_r = modify_points(x_inter_list.copy(), y_inter_list.copy(), seg_img_modify, offset=offset)
        for i, (line, rect) in enumerate(zip(lines, rects)):
            #print(line, horizon_line)
            x_anno, y_anno = get_inter_point(line[0], line[1], line[2], horizon_line[0], -1, horizon_line[1])
            if show:
                cv2.circle(seg_img_modify, (int(x_anno),int(y_anno)), 3, (0, 255, 0), thickness=2) 
            new_rect = modify_rect(rect, x_anno, y_anno)
            new_rects.append(new_rect)
            if show:   
                new_box_points = cv2.boxPoints(new_rect)
                new_box_points = np.int0(new_box_points)
                #draw contour里 x坐标在y前, 两列交换
                new_box_points = new_box_points[:, ::-1]
                cv2.drawContours(seg_img_modify, [new_box_points], 0, color_list[i], 2)
            
        if show:
            for line, x_anno in zip(lines, x_annos):
                #矩形代表直线在水平分割线上交点
                y_anno = (-line[2] + -line[0] * x_anno)/line[1]
                cv2.circle(seg_img_modify,  (int(x_anno), int(y_anno)), 3, (0, 255, 0), thickness=3 )
            plt_show(seg_img_modify)
        anchors = [rect2anchor(rect) for rect in new_rects]
        return [y_l, y_r, anchors]
    
#得到上顶点，x,y,w,h,theta(与x轴)
def get_anno_vp(frame_mp4_number, seg_width = 16, show = False, path2write = None, offset = 10):
    """
    display : 0, nothin; 1, show; 2, write
    return : (y1, y2, x1, w1, \theta)
    """
    #得到frame/mp4/number
    seg_img_file = os.path.join(seg_label_dir, frame_mp4_number+'.png')
    seg_img = cv2.imread(seg_img_file)
    H, W = seg_img.shape[:2]
    seg_img = seg_img * 50
    seg_img_ori = seg_img.copy()
    seg_img_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    #得到标注中有几条线，及其label
    seg_labels = np.sort(np.unique(seg_img_gray))
    #图中没有车道线
    if len(seg_labels) == 1:
        if(show):
            plt_show(seg_img)
        return None
        
    #保存直线参数的list,一条直线用3个参数表示[A，B, C]
    lines = []
    rects = []
    #得到覆盖标注的最小矩形
    for seg_label in seg_labels[1:]:
        #rect, box_points = get_min_rect_ori(seg_img_gray, seg_label)
        #cv2.drawContours(seg_img, [box_points], 0, (0, 255, 0), 2)
        try:
            rect, box_points = get_min_rect_v2(seg_img_gray, seg_label, seg_img)
        except:
            print(frame_mp4_number)
            REFINE_LIST.append(frame_mp4_number)
            continue
        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
        rects.append(rect)
        y_center = rect[0][0];
        x_center = rect[0][1];
        w=rect[1][0]
        h=rect[1][1]
#         w_anno = min(w, h)
#         h_anno = max(w, h)
        angle = rect[2]
        #求这个矩形代表的直线
        angle_y = angle_convert(w, h, angle)
        angle_x = angle_convert_x(w, h, angle)
        A, B, C = get_line_equation(y_center, x_center, angle_y)
        lines.append([A, B, C, angle_x])
    #plt_show(seg_img)
    #只有一条车道线的情况
    if(len(lines) == 1):
        if show:
            cv2.drawContours(seg_img, [box_points], 0, (0, 0, 255), 2)
            plt_show(seg_img)
        rect = rects[0]
        y_center = rect[0][0]
        x_center = rect[0][1]
        w=rect[1][0]
        h=rect[1][1]
        w_anno = min(w, h)
        h_anno = max(w, h)
        angle = rect[2] #+ 0.001
        #求这个矩形代表的直线
        angle_y = angle_convert(w, h, angle)
        angle_x = angle_convert_x(w, h, angle)
        #print(angle_y, angle_x)
        x_upper_center = x_center - h_anno/2 * np.cos(angle_x * np.pi / 180)
        y_upper_center = y_center - h_anno/2 * np.sin(angle_x * np.pi / 180)
        #anchor = rect2anchor(rects[0])
        return (int(round(x_upper_center)), int(round(y_upper_center)))
        
        
        
    #求交点
    x_inter_list=[]
    y_inter_list=[]
    for i in range(len(lines)-1):
        for j in range(i+1, len(lines)):
            #print(lines[i], lines[j])
            x_inter, y_inter = get_inter_point(lines[i][0], lines[i][1], lines[i][2], lines[j][0], lines[j][1], lines[j][2])
            x_inter_list.append(x_inter)
            y_inter_list.append(y_inter)

    return (int(round(np.mean(x_inter_list))), int(round(np.mean(y_inter_list))))


path2write = vp_label_dir 
if not os.path.exists(path2write):
    os.makedirs(path2write)

offset = 10
cnt = 0

def write_anno_vp(seg_img_file):
    global cnt
    with lock:
        cnt += 1
    if cnt %10000 == 0:
        print(cnt)
    try:
        name_split_list = seg_img_file.split('/')
        frame_mp4_number = '/'.join(name_split_list[-3:])[:-4]
        vp_img_path = os.path.join(path2write, frame_mp4_number+ '.png')
    #             if os.path.exists(txt_path):
    #                 continue
        dirname = os.path.dirname(vp_img_path )
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        annos = get_anno_vp(frame_mp4_number, seg_width=16, path2write = path2write, offset=10)
        vp_img = np.zeros((H, W), dtype=np.uint8)
        if annos is not None:
            cv2.circle(vp_img, annos, 16, 255, -1)
        vp_img[vp_img>0] = 1
        assert (vp_img>1).sum() == 0
        cv2.imwrite(vp_img_path, vp_img)
    except Exception as e:
        print(seg_img_file, "出现如下异常%s"%e, traceback.format_exc())

from multiprocessing.dummy import Pool, Lock
pool = Pool()
lock = Lock()
seg_img_list = glob(seg_label_dir + '/*/*/*png')
#write_anno_vp(seg_img_list)
pool.map(write_anno_vp, seg_img_list)
pool.close()
pool.join()
            
