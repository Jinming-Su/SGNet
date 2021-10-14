import torch
def cal_S(points):
    
    """
    #计算N个四边形的面积
    points:torch.Tensor() with shape (N, 4, 2)
    """
    x = points[..., 0]
    y = points[..., 1]
    x1 = x.roll(1, dims = 1)
    y1 = y.roll(1, dims = 1)
#     y1 = y[:, [1, 2, 3, 0]]
#     x1 = x[:, [1, 2, 3, 0]]
    return torch.sum((y*x1 - x*y1 )/2, 1)

def cal_norm_S(points, H, W):
    """
    代表y1, y2
    torch.Tensor((M, N, 2, 2))
    """
    #处理x一样的情况
    mask = (points[:, :, 0, 0] == points[:, :, 1, 0])
    y1 = (points[:, :, 1, 1] - points[:, :, 0, 1]) / (points[:, :, 1, 0] - points[:, :, 0, 0]) * (-points[:, :, 0, 0]) \
            + points[:, :, 0, 1]
    y2 = (points[:, :, 1, 1] - points[:, :, 0, 1]) / (points[:, :, 1, 0] - points[:, :, 0, 0]) * (W-1-points[:, :, 0, 0]) \
            + points[:, :, 0, 1]
    #print('y1, y2', y1, y2)
    return ((2*H-2-(y1 + y2)) * W / 2).masked_scatter_(mask, (2*H-2-(points[:, :, 0, 1] + points[:, :, 1, 1])) * W /2)
    #return (y1 + y2) * W /2 * mask + (points[:, :, 0, 1] + points[:, :, 1, 1]) * W /2 *~(mask)

            
def bbox_dis(a_bboxes, b_bboxes, pad_shape):
    """
    计算anchor和gt的面积量
    bboxes1: Tensor with shape (M, (x, y, w, a))
    bboxes2: Tensor with shape (N, (x, y, w, a))
    """
    H, W = pad_shape
    a_bboxes = a_bboxes.float()
    b_bboxes = b_bboxes.float()
    M = len(a_bboxes)
    N = len(b_bboxes)
    a_inter_p, a_mask1, a_mask2, a_mask3 = get_inter_with_border_mul(a_bboxes, H, W)
    b_inter_p, b_mask1, b_mask2, b_mask3 = get_inter_with_border_mul(b_bboxes, H, W)
    #print('inter_p', a_inter_p, b_inter_p)
    #(M, 2, 2) 2个点，2个坐标
    a_points = torch.stack((a_bboxes[:, :2], a_inter_p), dim=1)
    b_points = torch.stack((b_bboxes[:, :2], b_inter_p), dim=1)
    a_points = a_points.unsqueeze(1).repeat(1,N,1,1)
    #print('b_points', b_points)
    b_points = b_points.unsqueeze(0).repeat(M,1,1,1)
    #求用来归一化的面积
    #print(a_points.shape)
    norm_S = cal_norm_S(torch.stack((a_points[:, :, 0], b_points[:, :, 0]), dim=2), H, W)
    #print('norm_S', torch.nonzero(torch.lt(norm_S, 0)))
    points = torch.cat((a_points, b_points), dim=-2).reshape(-1, 4, 2)
    ord_points = order_points_tubao(points)
    #print('points', ord_points)
    S = cal_S(ord_points)
#     print('base_S', torch.nonzero(torch.lt(S, 0)))
#     print(S[torch.nonzero(torch.lt(S, 0))])
    #形成的梯形或三角形写成统一的形式
    add_S = torch.abs((H - 1 - a_points[..., 1, 1]  + H - 1 - b_points[..., 1, 1]) * (a_points[..., 1, 0] - b_points[..., 1, 0]) / 2)
    #print('add_S', add_S)   
    return (S.reshape(M, N) + add_S) / ( norm_S + 1e-6)
    
    
    
#     a_mask1.repeat(1, N)
#     a_mask2.repeat(1, N)
#     a_mask3.repeat(1, N)
#     b_mask1.repeat(M, 1)
#     b_mask2.repeat(M, 1)
#     b_mask3.repeat(M, 1)
#     #分别交于左右两边的情况：剩下的面积形成矩形或三角形
#     mask_add_rec = (a_mask1 & b_mask3) | (b_mask1 & a_mask3) 
#     mask_add_tri_l =  (a_mask1 & b_mask2) | (b_mask1 & a_mask2)
#     mask_add_tri_r =  (a_mask3 & b_mask2) | (b_mask3 & a_mask2)
#     print(mask_add_rec , mask_add_tri_l, mask_add_tri_r)
#     #交左右两边时时梯形
#     ret = ((H-1-a_points[..., 1, 1]) + (H-1 - b_points[..., 1, 1] )) * W/2
#     #左右下角的三角形
#     tri_l = torch.max(torch.abs(a_points[..., 1]-torch.Tensor([0, H-1])), dim = -1)[0] \
#         * torch.max(torch.abs(b_points[..., 1]-torch.Tensor([0, H-1])), dim = -1)[0] / 2
#     tri_r = torch.max(torch.abs(a_points[..., 1]-torch.Tensor([W-1, H-1])), dim = -1)[0] \
#         * torch.max(torch.abs(b_points[..., 1]-torch.Tensor([W-1, H-1])), dim = -1)[0] /2
#     S = S + mask_add_rec * ret + mask_add_tri_l * tri_l + mask_add_tri_r * tri_r
    #TODO checkcorrectness

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

#利用gram算法求凸包
def order_points_tubao(pts):
    """
    torch.Tensor (N, 4, 2)
    """
    pts = pts.float()
    #print('pts, ', pts)
    idx = torch.argsort(pts[..., 0], dim=-1)
    #TODO 更简洁的方法提取不同行的不同列
    x = torch.gather(pts[...,0], dim=1, index=idx)
    y = torch.gather(pts[...,1], dim=1, index=idx)
    x_sorted = torch.stack((x, y), dim=-1)
    #print('x_sorted', x_sorted)
    k = torch.clamp(torch.div(x_sorted[:, 1:,  1] - x_sorted[:, 0:1, 1], x_sorted[:, 1:,  0] - x_sorted[:, 0:1, 0]), \
                    -89, 89)
    
    #print('k', k)
    k_idx = torch.argsort(k, dim = -1)
    #print('k_idx', k_idx)
    #似乎只需要判断2，1与2，3的夹角
    k_sorted_x = torch.gather(x_sorted[:, 1:, 0], dim=-1, index = k_idx)
    k_sorted_y = torch.gather(x_sorted[:, 1:, 1], dim=-1, index = k_idx)
    mask_not_del2 = ( ( (k_sorted_x[:, 0] - k_sorted_x[:, 1]) * (k_sorted_y[:, 2] - k_sorted_y[:, 1]) - \
                (k_sorted_x[:, 2] - k_sorted_x[:, 1]) * (k_sorted_y[:, 0] - k_sorted_y[:, 1]) ) < 0 ).unsqueeze(-1)
    #print('mask_not_del2', mask_not_del2)
    #mask_not_del2 = torch.Tensor([True]).bool()
    #print(k_idx * mask_not_del2)
    idx = k_idx * mask_not_del2 + torch.cat((k_idx[:, [0, 2]], k_idx[:, -1:]), dim=-1) * (~mask_not_del2)
    #print('k_idx_reverse_1_2', k_idx_reverse_1_2)
    #print('idx', idx)
    x_rest = torch.gather(x_sorted[:, 1:, 0], dim=1, index = idx)
    y_rest = torch.gather(x_sorted[:, 1:, 1], dim=1, index = idx)
    rest  = torch.stack((x_rest, y_rest), dim=-1)
    ordered  = torch.cat((x_sorted[:, 0:1], rest), dim=1) 
    return ordered



#假设凸四边形
def order_points(pts):
    """
    torch.Tensor (N, 4, 2)
    """
    pts = pts.float()
    #print(pts)
    idx = torch.argsort(pts[..., 0], dim=-1)
    #TODO 更简洁的方法提取不同行的不同列
    x = torch.gather(pts[...,0], dim=1, index=idx)
    y = torch.gather(pts[...,1], dim=1, index=idx)
    x_sorted = torch.stack((x, y), dim=-1)
    #print('x_sorted', x_sorted)
    #print(x_sorted[:, 1:,  1] - x_sorted[:, 0:1, 1], x_sorted[:, 1:,  0] - x_sorted[:, 0:1, 0])
    k = torch.div(x_sorted[:, 1:,  1] - x_sorted[:, 0:1, 1], x_sorted[:, 1:,  0] - x_sorted[:, 0:1, 0])
    #print('k', k)
    k_idx = torch.argsort(k, dim = -1)
    #print('k_idx', k_idx)
    #最普通的情况，按顺序排列剩下3个点， 包含4个点在一条直线上
    
    
    #当第1、2个元素角度相同时，且2的距离离0更近的情况, 得到0，2，1，3
    #当3个点在一条垂直线上时，那么四个点就在一条直线上，所以不考虑
    #取出每一个rect的剩下三个点中的x元素，按角度的顺序排序
    x_sorted_1_2_3 = torch.gather(x_sorted[..., 1: , 0], dim=1, index=k_idx)
    #print('x_sorted_1_2_3', x_sorted_1_2_3)
    #float无法判定相等，取一个阈值
    mask_reverse_1_2 = (torch.abs(k[:, 0:1] - k[:, 1:2])<1e-3) & (x_sorted_1_2_3[:, 1:2] < x_sorted_1_2_3[:, 0:1])
    k_idx_reverse_1_2 = k_idx[..., [1, 0, 2]]
    #当第1、2个元素不同，第2，3个元素角度相同时，且3的距离离0更近的情况, 得到0，1，4，3
    #print(torch.abs(k[:, 1:2] - k[:, 2:3])<1e-3, (x_sorted_1_2_3[:, 2:3] < x_sorted_1_2_3[: ,1:2]))
    mask_reverse_2_3 = ~(mask_reverse_1_2) & (torch.abs(k[:, 1:2] - k[:, 2:3])<1e-3) & (x_sorted_1_2_3[:, 1:2] < x_sorted_1_2_3[: ,2:3])
    #print('mask', (~mask_reverse_1_2) & (~mask_reverse_2_3), mask_reverse_1_2, mask_reverse_2_3)
    k_idx_reverse_2_3 = k_idx[..., [0, 2, 1]]
    #print('k_idx', k_idx)
    idx = k_idx * ((~mask_reverse_1_2) & (~mask_reverse_2_3)) + mask_reverse_1_2 * k_idx_reverse_1_2 + \
          mask_reverse_2_3 * k_idx_reverse_2_3
    #print('k_idx_reverse_1_2', k_idx_reverse_1_2)
    #print('idx', idx)
    x_rest = torch.gather(x_sorted[:, 1:, 0], dim=1, index = idx)
    y_rest = torch.gather(x_sorted[:, 1:, 1], dim=1, index = idx)
    rest  = torch.stack((x_rest, y_rest), dim=-1)
    ordered  = torch.cat((x_sorted[:, 0:1, ], rest), dim=1)
    #print('ordered', ordered)
    
#     for rect in ordered:     
#         draw_polygon(rect.cpu().numpy())
    return torch.cat((x_sorted[:, 0:1, ], rest), dim=1)



def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    return ious
