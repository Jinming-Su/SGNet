import numpy as np
import torch
import numpy as np


def cal_S(points):
    """
    #计算N个四边形的面积
    points:np.array with shape (N, 4, 2)
    """
    x = points[..., 0]
    y = points[..., 1]
    x1 = np.roll(x, 1, axis=1)
    y1 = np.roll(y, 1, axis=1)
    #     y1 = y[:, [1, 2, 3, 0]]
    #     x1 = x[:, [1, 2, 3, 0]]
    return np.sum((y * x1 - x * y1) / 2, 1)


def cal_norm_S(points, H, W):
    """
    代表y1, y2
    np.array((M, N, 2, 2))
    """
    mask = (points[:, :, 0, 0] == points[:, :, 1, 0])
    y1 = np.clip( (points[:, :, 1, 1] - points[:, :, 0, 1]) / (points[:, :, 1, 0] - points[:, :, 0, 0]) * (-points[:, :, 0, 0]) \
            + points[:, :, 0, 1] , 0, H-1)
    y2 = np.clip( (points[:, :, 1, 1] - points[:, :, 0, 1]) / (points[:, :, 1, 0] - points[:, :, 0, 0]) * (W-1-points[:, :, 0, 0]) \
            + points[:, :, 0, 1] , 0, H-1)
    return (2*H-2-(y1 + y2)) * W / 2
    #return (y1 + y2) * W /2 * mask + (points[:, :, 0, 1] + points[:, :, 1, 1]) * W /2 *~(mask)

def bbox_dis(a_bboxes, b_bboxes, pad_shape=(590, 1640)):
    """
    计算anchor和gt的面积量
    bboxes1: array with shape (M, (x, y, w, a))
    bboxes2: array with shape (N, (x, y, w, a))
    """
    H, W = pad_shape
    a_bboxes = a_bboxes.astype(np.float)
    b_bboxes = b_bboxes.astype(np.float)
    M = len(a_bboxes)
    N = len(b_bboxes)
    a_inter_p, a_mask1, a_mask2, a_mask3 = get_inter_with_border_mul(a_bboxes, H, W)
    b_inter_p, b_mask1, b_mask2, b_mask3 = get_inter_with_border_mul(b_bboxes, H, W)
    # print('inter_p', a_inter_p, b_inter_p)
    # (M, 2, 2) 2个点，2个坐标
    a_points = np.stack((a_bboxes[:, :2], a_inter_p), axis=1)
    b_points = np.stack((b_bboxes[:, :2], b_inter_p), axis=1)
    a_points = np.tile(np.expand_dims(a_points, 1), (1, N, 1, 1))
    # print('b_points', b_points)
    b_points = np.tile(np.expand_dims(b_points, 0), (M, 1, 1, 1))
    # 求用来归一化的面积
    # print(a_points.shape)
    norm_S = cal_norm_S(np.stack((a_points[:, :, 0], b_points[:, :, 0]), axis=2), H, W)
    # print('norm_S', torch.nonzero(torch.lt(norm_S, 0)))
    points = np.concatenate((a_points, b_points), axis=-2).reshape(-1, 4, 2)
    ord_points = order_points_tubao(points)
    # print('points', ord_points)
    S = cal_S(ord_points)
    #     print('base_S', torch.nonzero(torch.lt(S, 0)))
    #     print(S[torch.nonzero(torch.lt(S, 0))])
    # 形成的梯形或三角形写成统一的形式
    add_S = np.abs(
        (H - 1 - a_points[..., 1, 1] + H - 1 - b_points[..., 1, 1]) * (a_points[..., 1, 0] - b_points[..., 1, 0]) / 2)
    # print('add_S', add_S)
    return (S.reshape(M, N) + add_S) / (norm_S + 1e-6)


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
# TODO checkcorrectness

def get_inter_with_border_mul(bboxes, H, W):
    """
    并行得到N个anchor与边界的交点
    bboxes : np.array (N, 4) in (x, y, w, a)
    """
    # bboxes = bboxes.float()
    k1 = (H - 1 - bboxes[:, 1]) / (-bboxes[:, 0] + 1e-6)
    k2 = (H - 1 - bboxes[:, 1]) / (W - 1 - bboxes[:, 0] + 1e-6)
    k = np.tan(bboxes[:, 3] * np.pi / 180)
    #print('k_in_border', k)
    mask1 = ((bboxes[:, 3] > 90) & (k >= k1)).reshape((-1, 1))
    mask3 = ((bboxes[:, 3] <= 90) & (k <= k2)).reshape((-1, 1))
    mask2 = (~(mask1 | mask3)).reshape((-1, 1))
    #print('mask', mask1, mask2, mask3)
    # 左边交点的y
    p_l = np.zeros_like(bboxes[:, :2])
    p_d = np.zeros_like(p_l)
    p_r = np.zeros_like(p_l)
    p_l[:, 1] = -k * bboxes[:, 0] + bboxes[:, 1]
    # 下边交点的x
    p_d[:, 1].fill(H - 1)
    p_d[:, 0] = (H - 1 - bboxes[:, 1]) / (k + 1e-6) + bboxes[:, 0]
    # 右边交点的y
    p_r[:, 0].fill(W - 1)
    p_r[:, 1] = k * (W - 1 - bboxes[:, 0]) + bboxes[:, 1]
    inter_p = mask1 * p_l + mask2 * p_d + mask3 * p_r
    return inter_p, mask1, mask2, mask3


# 利用gram算法求凸包
def order_points_tubao(pts):
    """
    np.array (N, 4, 2)
    """

    idx = np.argsort(pts[..., 0], axis=-1)
    # TODO 更简洁的方法提取不同行的不同列
    x = gather(pts[..., 0], dim=1, index=idx)
    y = gather(pts[..., 1], dim=1, index=idx)
    x_sorted = np.stack((x, y), axis=-1)
    #print('x_sorted', x_sorted)
    k = np.clip( (x_sorted[:, 1:, 1] - x_sorted[:, 0:1, 1]) / (x_sorted[:, 1:, 0] - x_sorted[:, 0:1, 0]), \
                    -89, 89)
    # print('k', k)
    k_idx = np.argsort(k, axis=-1)
    # print('k_idx', k_idx)
    # 似乎只需要判断2，1与2，3的夹角
    k_sorted_x = gather(x_sorted[:, 1:, 0], -1, k_idx)
    k_sorted_y = gather(x_sorted[:, 1:, 1], -1, k_idx)
    mask_not_del2 = np.expand_dims((((k_sorted_x[:, 0] - k_sorted_x[:, 1]) * (k_sorted_y[:, 2] - k_sorted_y[:, 1]) - \
                      (k_sorted_x[:, 2] - k_sorted_x[:, 1]) * (k_sorted_y[:, 0] - k_sorted_y[:, 1])) < 0), -1)
    # print('mask_not_del2', mask_not_del2)
    # mask_not_del2 = torch.Tensor([True]).bool()
    # print(k_idx * mask_not_del2)
    idx = k_idx * mask_not_del2 + np.concatenate((k_idx[:, [0, 2]], k_idx[:, -1:]), axis=-1) * (~mask_not_del2)
    # print('k_idx_reverse_1_2', k_idx_reverse_1_2)
    # print('idx', idx)
    x_rest = gather(x_sorted[:, 1:, 0], 1, idx)
    y_rest = gather(x_sorted[:, 1:, 1], 1, idx)
    rest = np.stack((x_rest, y_rest), axis=-1)
    ordered = np.concatenate((x_sorted[:, 0:1], rest), axis=1)
    return ordered

def gather(self, dim, index):
    """
    Gathers values along an axis specified by ``dim``.

    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters
    ----------
    dim:
        The axis along which to index
    index:
        A tensor of indices of elements to gather

    Returns
    -------
    Output Tensor
    """
    idx_xsection_shape = index.shape[:dim] + \
        index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def check_correctness():
    #均交于下边、
    print(bbox_dis(np.array([[600, 500, 16, 90]]), np.array([[400, 500, 16, 90]])), '\n')
    #交叉的下边
    print(bbox_dis(np.array([[600, 500, 16, 45]]), np.array([[400, 500, 16, 135]])), '\n')
    #交于左右两边
    #print(bbox_dis(torch.tensor([[600, 500, 16, 0]]), torch.tensor([[400, 500, 16, 180]])), '\n')
    #print(bbox_dis(torch.tensor([[600, 500, 16, 30]]), torch.tensor([[400, 500, 16, 150]])), '\n')
    #交于1左1下
    #print(bbox_dis(torch.tensor([[600, 500, 16, 90]]), torch.tensor([[400, 500, 16, 150]])), '\n')
    #print(bbox_dis(torch.tensor([[600, 500, 16, 0]]), torch.tensor([[400, 500, 16, 40]])), '\n')
    #交于1右1下
    #print(bbox_dis(torch.tensor([[600, 500, 16, 90]]), torch.tensor([[400, 500, 16, 30]])), '\n')
    #x 相同 测试norm_S
    #print(bbox_dis(torch.tensor([[400, 500, 16, 90]]), torch.tensor([[400, 500, 16, 30]])), '\n')
    #print(bbox_dis(torch.tensor([[600, 500, 16, 60]]), torch.tensor([[590, 510, 16, 70]])), '\n')
    #测试3点重合时
    #print(bbox_dis(torch.tensor([[600, 500, 16, 90]]), torch.tensor([[400, 500, 16, 180]])), '\n')
    #print(bbox_dis(torch.tensor([[600, 500, 16, 60]]), torch.tensor([[400, 0, 16, 180]])), '\n')
    #测试第0，2、3个点一条线
    print(bbox_dis(np.array([[600, 500, 16, 60]]), np.array([[400, 589, 16, 0]])), '\n')
    pass
