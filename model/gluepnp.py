import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from copy import deepcopy
from torch.autograd import grad
import utilities.geometry_utilities as geo
import torch.utils.data as Data
from lib.losses import *
from .epropnp.epropnp import EProPnP6DoF
from .epropnp.levenberg_marquardt import LMSolver, RSLMSolver
from .epropnp.camera import PerspectiveCamera
from .epropnp.cost_fun import AdaptiveHuberPnPCost


# ops  import conv1d_layer_knnGraph, conv1d_resnet_block_knnGraph
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=10, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2).contiguous()  # (batch_size, num_dims*2, num_points, k)
    return feature


def bn_act(in_channel, perform_gcn, perform_bn, activation):
    """
    # Global Context normalization on the input
    """
    layers = []
    if perform_gcn:
        layers.append(gcn(in_channel))

    if perform_bn:
        layers.append(nn.BatchNorm1d(in_channel, affine=False))

    if activation == 'relu':
        layers.append(torch.nn.ReLU())

    return layers


def conv1d_layer_knnGraph(in_channel, out_channel, ksize, activation, perform_bn, perform_gcn, nb_neighbors=10,
                          act_pos="post"):
    assert act_pos == "pre" or act_pos == "post" or act_pos == "None"
    layers = []
    # If pre activation
    if act_pos == "pre":
        new = bn_act(in_channel, perform_gcn, perform_bn, activation)
        for l in new:
            layers.append(l)

    if nb_neighbors > 0:
        # get the knn graph features
        layers.append(knn_feature(nb_neighbors, in_channel, out_channel, ksize))
    else:
        # no knn graph here, only MLP at per-point
        layers.append(torch.nn.Conv1d(in_channel, out_channel, ksize))

    # If post activation
    if act_pos == "post":

        new = bn_act(out_channel, perform_gcn, perform_bn, activation)
        for l in new:
            layers.append(l)

    return nn.Sequential(*layers)


class gcn(nn.Module):
    def __init__(self, in_channel):
        super(gcn, self).__init__()
        pass

    def forward(self, x):
        # x: [n, c, K]
        var_eps = 1e-3
        m = torch.mean(x, 2, keepdim=True)
        v = torch.var(x, 2, keepdim=True)
        inv = 1. / torch.sqrt(v + var_eps)
        x = (x - m) * inv
        return x


class knn_feature(nn.Module):
    def __init__(self, nb_neighbors, in_channel, out_channel, ksize):
        super(knn_feature, self).__init__()
        self.k = nb_neighbors
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ksize = ksize
        self.conv = torch.nn.Conv2d(in_channel * 2, out_channel, ksize)

    def forward(self, x):
        # x: [n, c, K]
        x = get_graph_feature(x, k=self.k)
        # conv
        x = self.conv(x)
        # avg pooling
        x = x.mean(dim=-1, keepdim=False)

        return x


class conv1d_resnet_block_knnGraph(nn.Module):
    def __init__(self, in_channel, out_channel, ksize, activation, perform_bn=False, perform_gcn=False, nb_neighbors=10,
                 act_pos="post"):
        super(conv1d_resnet_block_knnGraph, self).__init__()

        # Main convolution
        self.conv1 = conv1d_layer_knnGraph(
            in_channel=in_channel,
            out_channel=out_channel,
            ksize=ksize,
            activation=activation,
            perform_bn=perform_bn,
            perform_gcn=perform_gcn,
            nb_neighbors=nb_neighbors,
            act_pos=act_pos
        )

        # Main convolution
        self.conv2 = conv1d_layer_knnGraph(
            in_channel=out_channel,
            out_channel=out_channel,
            ksize=ksize,
            activation=activation,
            perform_bn=perform_bn,
            perform_gcn=perform_gcn,
            nb_neighbors=nb_neighbors,
            act_pos=act_pos
        )

    def forward(self, x):
        xorg = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + xorg


def ransac_p3p(P, p2d, p3d, num_points_2d, num_points_3d):
    # 1. Choose top k correspondences:
    k = min(1000, round(1.5 * p2d.size(-2)))  # Choose at most 1000 points
    P_value, P_topk_i = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)
    p2d_indices = P_topk_i // P.size(-1)  # bxk (integer division)
    p3d_indices = P_topk_i % P.size(-1)  # bxk
    K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
    theta0 = P.new_zeros((P.size(0), 6))

    # p2d_mink = []
    # p3d_mink = []
    # P_topk = []
    # 2. Loop over batch and run RANSAC:
    for i in range(P.size(0)):
        num_points_ransac = min(k, round(1.5 * num_points_2d[i].float().item()),
                                round(1.5 * num_points_3d[i].float().item()))
        num_points_ransac = min(k, max(num_points_ransac, 10))  # At least 10 points
        p2d_np = p2d[i, p2d_indices[i, :num_points_ransac], :].cpu().numpy()
        p3d_np = p3d[i, p3d_indices[i, :num_points_ransac], :].cpu().numpy()
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            p3d_np, p2d_np, K, dist_coeff,
            iterationsCount=1000,
            reprojectionError=0.01,
            confidence=0.99,
            flags=cv2.SOLVEPNP_P3P)
        # print(inliers.shape[0], '/',  num_points_2d[i].item())
        if rvec is not None and tvec is not None and retval:

            rvec = torch.as_tensor(rvec, dtype=P.dtype, device=P.device).squeeze(-1)
            tvec = torch.as_tensor(tvec, dtype=P.dtype, device=P.device).squeeze(-1)
            if torch.isfinite(rvec).all() and torch.isfinite(tvec).all():
                theta0[i, :3] = rvec
                theta0[i, 3:] = tvec
            # p2d_mink.append(p2d[i, p2d_indices[i, :len(inliers)], :])
            # p3d_mink.append(p3d[i, p3d_indices[i, :len(inliers)], :])
            # P_topk.append(P_value[i, :len(inliers)])

    return theta0  # p2d_mink, p3d_mink, P_topk


# gnn
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


# feature extraction for 3D and 2D points cloud
class FeatureExtractor(nn.Module):
    def __init__(self, in_channel):

        super(FeatureExtractor, self).__init__()

        activation = 'relu'
        idx_layer = 0
        self.numlayer = 2
        ksize = 1
        nchannel = 128
        act_pos = "post"
        knn_nb = 10

        conv1d_block = conv1d_resnet_block_knnGraph
        # First convolution
        # just used to change the dim of in_chan to nchannel
        self.conv_in = conv1d_layer_knnGraph(
            in_channel=in_channel,
            out_channel=nchannel,
            ksize=ksize,
            activation=None,
            perform_bn=False,
            perform_gcn=False,
            nb_neighbors=knn_nb,
            act_pos="None"
        )
        # ResNet Knn graph
        for _ksize, _nchannel in zip([ksize] * self.numlayer, [nchannel] * self.numlayer):
            setattr(self, 'conv_%d' % idx_layer, conv1d_block(
                in_channel=nchannel,
                out_channel=nchannel,
                ksize=_ksize,
                activation=activation,
                perform_bn=True,
                perform_gcn=True,
                nb_neighbors=knn_nb,
                act_pos=act_pos
            ))

            idx_layer += 1

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.numlayer):
            x = getattr(self, 'conv_%d' % i)(x)
        return x


# a small T-net to transform 3D points to a cano. direction
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
        x = x.view(-1, 3, 3) + iden
        return x


# calculate the pairwise distance for 3D and 2D features
def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets
    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res


# Sinkhorn to estimate the joint probability matrix P
class sinkhorn(torch.nn.Module):
    def __init__(self, mu=0.1, tolerance=1e-9, iterations=20):
        super(sinkhorn, self).__init__()
        # self.config = config
        self.mu = mu  # the smooth term
        self.tolerance = tolerance  # don't change
        self.iterations = iterations  # max 30 is set, enough for a typical sized mat (e.g., 1000x1000)
        self.eps = 1e-12

    def forward(self, x2d, x3d, num_points_2d, num_points_3d):
        x2d_norm2 = x2d.pow(2).sum(dim=-1, keepdim=True)
        x3d_norm2 = x3d.pow(2).sum(dim=-1, keepdim=True)
        M = torch.baddbmm(
            x3d_norm2.transpose(-2, -1),
            x2d,
            x3d.transpose(-2, -1),
            alpha=-2
        ).add_(x2d_norm2).clamp_min_(1e-30).sqrt_()

        b, m, n = M.size()
        new_M = torch.zeros(b, m + 1, n + 1, device=M.device)
        # 将原始张量复制到新张量的适当位置
        new_M[:, :m, :n] = M
        m, n = m+1, n+1

        r = new_M.new_zeros((b, m))  # bxm
        c = new_M.new_zeros((b, n))  # bxn
        for i in range(b):
            r[i, :(num_points_2d[i]+1)] = 1.0 / (num_points_2d[i].float() + 1.0)
            c[i, :(num_points_3d[i]+1)] = 1.0 / (num_points_3d[i].float() + 1.0)

        # r, c are the prior 1D prob distribution of 3D and 2D points, respectively
        # M is feature distance between 3D and 2D point
        K = (-new_M / self.mu).exp()
        # 1. normalize the matrix K
        K = K / K.sum(dim=(-2, -1), keepdim=True).clamp_min_(self.eps)  # 同M 8 150 150

        # 2. construct the unary prior

        r = r.unsqueeze(-1)  # 8 150 1
        u = r.clone()  # 8 150 1
        c = c.unsqueeze(-1)  # 8 150 1

        i = 0
        u_prev = torch.ones_like(u)
        while (u - u_prev).norm(dim=-1).max() > self.tolerance:
            if i > self.iterations:
                break
            i += 1
            u_prev = u
            # update the prob vector u, v iteratively
            v = c / K.transpose(-2, -1).matmul(u).clamp_min_(self.eps)
            u = r / K.matmul(v).clamp_min_(self.eps)

        # assemble
        # P = torch.diag_embed(u[:,:,0]).matmul(K).matmul(torch.diag_embed(v[:,:,0]))
        P = (u * K) * v.transpose(-2, -1)
        return P


class getPoseInit(nn.Module):
    def __init__(self,epropnp, camera=PerspectiveCamera(),
                 cost_fun=AdaptiveHuberPnPCost(relative_delta=0.5)):
        super().__init__()
        self.epropnp = epropnp
        self.camera = camera
        self.cost_fun = cost_fun

    def forward(self, P, p2d, p3d, num_points_2d, num_points_3d, ransac=True):
        # At the beginning of training, it is recommended to use RANSAC
        # When the corresponding loss is relatively small, it can be switched to epropnp
        if ransac:
            # 1. Choose top k correspondences:
            k = min(800, round(0.9 * p2d.size(-2)))  # Choose at most 1000 points
            _, P_topk_i = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)
            p2d_indices = P_topk_i // P.size(-1)  # bxk (integer division)
            p3d_indices = P_topk_i % P.size(-1)  # bxk
            K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
            dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
            theta0 = P.new_zeros((P.size(0), 6))
            # 2. Loop over batch and run RANSAC:
            for i in range(P.size(0)):
                num_points_ransac = min(k, round(1.5 * num_points_2d[i].float().item()),
                                        round(1.5 * num_points_3d[i].float().item()))
                num_points_ransac = min(k, max(num_points_ransac, 10))  # At least 10 points
                p2d_np = p2d[i, p2d_indices[i, :num_points_ransac], :].cpu().numpy()
                p3d_np = p3d[i, p3d_indices[i, :num_points_ransac], :].cpu().numpy()
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    p3d_np, p2d_np, K, dist_coeff,
                    iterationsCount=1000,
                    reprojectionError=0.01,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_P3P)
                # print(inliers.shape[0], '/',  num_points_2d[i].item())
                if rvec is not None and tvec is not None and retval:
                    rvec = torch.as_tensor(rvec, dtype=P.dtype, device=P.device).squeeze(-1)
                    tvec = torch.as_tensor(tvec, dtype=P.dtype, device=P.device).squeeze(-1)
                    if torch.isfinite(rvec).all() and torch.isfinite(tvec).all():
                        theta0[i, :3] = rvec
                        theta0[i, 3:] = tvec
        else:
            top_k = min(200, round(0.8 * p2d.size(-2)))
            # top_k = min(100, round(0.6 * num_points_2d[0].float().item()))
            # if top_k < 10:
            #     top_k = num_points_2d[0].float().item()
            top_k = int(top_k)
            P_value, P_topk_i = torch.topk(P.flatten(start_dim=-2), k=top_k, dim=-1, largest=True, sorted=True)
            p2d_indices = P_topk_i // P.size(-1)  # bxk (integer division)
            p3d_indices = P_topk_i % P.size(-1)  # bxk
            p2d_sort = p2d[torch.arange(p2d.shape[0]).unsqueeze(1), p2d_indices]
            p3d_sort = p3d[torch.arange(p3d.shape[0]).unsqueeze(1), p3d_indices]
            P_sort = P_value[:, :top_k].unsqueeze(2)

            cam_mats = torch.eye(3, dtype=torch.float32, device=p3d.device)
            batch_cam_mats = cam_mats.expand(1, -1, -1)
            self.camera.set_param(batch_cam_mats)
            theta0 = []
            for i in range(P.size(0)):
                # num_points_ransac = min(100, 0.6*num_points_2d[i].item(), 0.6*num_points_3d[i].item())
                # num_points_ransac = int(min(100, max(int(0.6*num_points_ransac), 10)))
                num_points_ransac = 50
                p2d_min = p2d_sort[i, :num_points_ransac, :].unsqueeze(0)
                p3d_min = p3d_sort[i, :num_points_ransac, :].unsqueeze(0)
                P_min = P_sort[i, :num_points_ransac, :].unsqueeze(0)
                P_min = P_min * min(num_points_2d[i].float().item(), num_points_3d[i].float().item())
                P_min_replicated = P_min.clone()
                w2d = torch.cat([P_min, P_min_replicated], dim=-1)
                self.cost_fun.set_param(p2d_min.detach(), w2d)  # compute dynamic delta
                theta, _, _, _ = self.epropnp(
                    p3d_min, p2d_min, w2d, self.camera, self.cost_fun,
                    fast_mode=False)  # fast_mode=True activates Gauss-Newton solver (no trust region)
                theta0.append(theta)
            theta0 = torch.cat(theta0, dim=0)

        return theta0


def generate_random_3d_points(num_points, x_range, y_range, z_range):
    # 创建一个包含随机三维坐标的数组
    points = np.random.rand(num_points, 3)
    # 缩放坐标到指定范围
    points *= np.array([x_range, y_range, z_range + 1.0])
    return points


def weightedReprojectionErrorwithPose(W, p2d, p3d, pose_quat):
    """ Weighted angular reprojection error

    f(W, p2d, p3d, theta) = sum_{i=1}^m sum_{j=1}^n w_ij (1 - p2d_i^T N(R(theta) p3d_j + t(theta)))
        N(p3d) = p3d / ||p3d||

    Arguments:
        W: (b, m*n) Torch tensor,
            batch of flattened weight matrices

        p2d: (b, m, 3) Torch tensor,
            batch of 3D bearing-vector-sets,
            assumption: unit norm

        p3d: (b, n, 3) Torch tensor,
            batch of 3D point-sets

        theta: (b, 6) Torch tensor,
            batch of transformation parameters
            assumptions:
                theta[:, 0:3]: angle-axis rotation vector
                theta[:, 3:6]: translation vector

    Return Values:
        error: (b, ) Torch tensor,
            sum cosine "distance"

    Complexity:
        O(bmn)
    """
    b = p2d.size(0)
    m = p2d.size(1)
    n = p3d.size(1)
    W = W.reshape(b, m, n)
    if pose_quat.shape[1]==7:
        rr = geo.quaternion_to_matrix(pose_quat[..., 3:])
        tt = pose_quat[..., :3]
    else:
        rr = geo.angle_axis_to_rotation_matrix(pose_quat[..., :3])
        tt = pose_quat[..., 3:]
    p3dt = geo.transform_and_normalise_points_by_Rt(p3d, rr, tt)
    error = torch.einsum('bmn,bmn->b', (W, 1.0 - torch.einsum('bmd,bnd->bmn', (p2d, p3dt))))
    return error


def Dy(f, x, y):
    """
    Dy(x) = -(D_YY^2 f(x, y))^-1 D_XY^2 f(x, y)
    Lemma 4.3 from
    Stephen Gould, Richard Hartley, and Dylan Campbell, 2019
    "Deep Declarative Networks: A New Hope", arXiv:1909.04866

    Arguments:
        f: (b, ) Torch tensor, with gradients
            batch of objective functions evaluated at (x, y)

        x: (b, n) Torch tensor, with gradients
            batch of input vectors

        y: (b, m) Torch tensor, with gradients
            batch of minima of f

    Return Values:
        Dy(x): (b, m, n) Torch tensor,
            batch of gradients of y with respect to x

    """
    with torch.enable_grad():
        grad_outputs = torch.ones_like(f)
        DYf = grad(f, y, grad_outputs=grad_outputs, create_graph=True)[0]  # bxm
        DYYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, y.size(-1)))  # bxmxm
        DXYf = torch.empty_like(DYf.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # bxmxn
        grad_outputs = torch.ones_like(DYf[:, 0])
        for i in range(DYf.size(-1)):  # [0,m)
            DYf_i = DYf[:, i]  # b
            DYYf[:, i:(i + 1), :] = grad(DYf_i, y, grad_outputs=grad_outputs, create_graph=True)[
                0].contiguous().unsqueeze(1)  # bx1xm
            DXYf[:, i:(i + 1), :] = grad(DYf_i, x, grad_outputs=grad_outputs, create_graph=True)[
                0].contiguous().unsqueeze(1)  # bx1xn
    DYYf = DYYf.detach()
    DXYf = DXYf.detach()
    DYYf = 0.5 * (DYYf + DYYf.transpose(1, 2))  # In case of floating point errors

    # Try a batchwise solve, otherwise revert to looping
    # Avoids cuda runtime error (9): invalid configuration argument
    try:
        U = torch.cholesky(DYYf, upper=True)
        Dy_at_x = torch.cholesky_solve(-1.0 * DXYf, U, upper=True)  # bxmxn
    except:
        Dy_at_x = torch.empty_like(DXYf)
        for i in range(DYYf.size(0)):  # For some reason, doing this in a loop doesn't crash
            try:
                U = torch.cholesky(DYYf[i, ...], upper=True)
                Dy_at_x[i, ...] = torch.cholesky_solve(-1.0 * DXYf[i, ...], U, upper=True)
            except:
                Dy_at_x[i, ...], _ = torch.solve(-1.0 * DXYf[i, ...], DYYf[i, ...])

    # Set NaNs to 0:
    if torch.isnan(Dy_at_x).any():
        Dy_at_x[torch.isnan(Dy_at_x)] = 0.0  # In-place
    # Clip gradient norms:
    max_norm = 100.0
    Dy_norm = Dy_at_x.norm(dim=-2, keepdim=True)  # bxmxn
    if (Dy_norm > max_norm).any():
        clip_coef = (max_norm / (Dy_norm + 1e-6)).clamp_max_(1.0)
        Dy_at_x = clip_coef * Dy_at_x

    return Dy_at_x

class LBFGSBlindPnPFn(torch.autograd.Function):
    """
    A class to optimise the weighted angular reprojection error given
    a set of 3D point p3d, a set of bearing vectors p2d, a weight matrix W
    """

    @staticmethod
    def forward(ctx, W, p2d, p3d, theta0=None):
        """ Optimise the weighted angular reprojection error

        Arguments:
            W: (b, m, n) Torch tensor,
                batch of weight matrices,
                assumption: positive and sum to 1 per batch

            p2d: (b, m, 3) Torch tensor,
                batch of 3D bearing-vector-sets,
                assumption: unit norm

            p3d: (b, n, 3) Torch tensor,
                batch of 3D point-sets

            theta0: (b, 6) Torch tensor,
                batch of initial transformation parameters
                assumptions:
                    theta[:, 0:3]: angle-axis rotation vector
                    theta[:, 3:6]: translation vector

        Return Values:
            theta: (b, 6) Torch tensor,
                batch of optimal transformation parameters
        """
        W = W.detach()
        p2d = p2d.detach()
        p3d = p3d.detach()
        if theta0 is None:
            theta0 = W.new_zeros((W.size()[0], 7))
        theta0 = theta0.detach()
        W = W.flatten(start_dim=-2)

        # Use a variable maximum number of iterations (aim: ~1s per pair)
        max_num_points = max(p2d.size(-2), p3d.size(-2))
        max_iter = round(max(min(100, 75 * pow(max_num_points / 1000.0, -1.5)), 1))

        with torch.enable_grad():
            theta = theta0.clone().requires_grad_(True)  # bx7
            # Note: in batch-mode, stopping conditions are entangled across batches
            # It would be better to use group norms for the stopping conditions
            opt = torch.optim.LBFGS([theta],
                                    lr=1.0,  # Default: 1
                                    max_iter=max_iter,  # Default: 100
                                    max_eval=None,
                                    tolerance_grad=1e-40,  # Default: 1e-05
                                    tolerance_change=1e-40,  # Default: 1e-09
                                    history_size=100,
                                    line_search_fn="strong_wolfe",
                                    )

            def closure():
                if torch.is_grad_enabled():
                    opt.zero_grad()
                error = weightedReprojectionErrorwithPose(W, p2d, p3d, theta).mean()  # average over batch
                if error.requires_grad:
                    error.backward()
                return error

            opt.step(closure)
        theta = theta.detach()
        ctx.save_for_backward(W, p2d, p3d, theta)
        return theta.clone()

    @staticmethod
    def backward(ctx, grad_output):
        W, p2d, p3d, theta = ctx.saved_tensors
        b = p2d.size(0)
        m = p2d.size(1)
        n = p3d.size(1)
        grad_input = None
        if ctx.needs_input_grad[0]:  # W only
            with torch.enable_grad():
                W = W.detach().requires_grad_()
                theta = theta.detach().requires_grad_()
                fn_at_theta = weightedReprojectionErrorwithPose(W, p2d, p3d, theta)  # b
            Dtheta = Dy(fn_at_theta, W, theta)  # bx6xmn
            grad_input = torch.einsum("ab,abc->ac", (grad_output, Dtheta))  # bx6 * bx6xmn-> bxmn
            grad_input = grad_input.reshape(b, m, n)
        return grad_input, None, None, None


class LBFGSBlindPnP(torch.nn.Module):
    def __init__(self):
        super(LBFGSBlindPnP, self).__init__()

    def forward(self, W, p2d, p3d, pose_quat):
        return LBFGSBlindPnPFn.apply(W, p2d, p3d, pose_quat)


class DeepPnP(nn.Module):
    def __init__(self, epropnp0):
        super().__init__()
        self.in_channel_2d = 2  # normalize 2d points
        self.in_channel_3d = 3  # X,Y,Z coordinates of 3D points
        self.stn = STN3d()  # a small T-net to transform 3D points to a cano. direction
        # feature extractors for 3D and 2D branch
        self.FeatureExtractor2d = FeatureExtractor(self.in_channel_2d)
        self.FeatureExtractor3d = FeatureExtractor(self.in_channel_3d)

        self.pairwiseL2Dist = pairwiseL2Dist
        # configurations for the estimation of joint probability matrix
        self.sinkhorn_mu = 0.1
        self.sinkhorn_tolerance = 1e-9
        self.iterations = 30
        self.sinkhorn = sinkhorn(self.sinkhorn_mu, self.sinkhorn_tolerance, self.iterations)
        self.dim = 128
        self.gnn_layers = ['self', 'cross'] * 2
        self.gnn = AttentionalGNN(
            self.dim, self.gnn_layers)
        self.cov1d = nn.Conv1d(128, 128, kernel_size=1, bias=True)

        # self.ransac_p3p = ransac_p3p
        # self.wbpnp = NonlinearWeightedBlindPnP()

        self.min_epropnp = getPoseInit(epropnp0)
        self.solvepnp = LBFGSBlindPnP()


    def forward(self, p2d, p3d, num_points_2d, num_points_3d, pose=None, poseloss=True):
        f3d = p3d
        f2d = p2d
        # Transform f3d to canonical coordinate frame:
        trans = self.stn(f3d.transpose(-2, -1))  # bx3x3
        f3d = torch.bmm(f3d, trans)  # bxnx3
        # Extract features

        f2d = self.FeatureExtractor2d(f2d.transpose(-2, -1))
        f3d = self.FeatureExtractor3d(f3d.transpose(-2, -1))

        f3d, f2d = self.gnn(f3d, f2d)

        f3d = self.cov1d(f3d)
        f2d = self.cov1d(f2d)

        mf3d, mf2d = f3d.transpose(-2, -1), f2d.transpose(-2, -1)

        # L2 Normalise:
        mf2d = torch.nn.functional.normalize(mf2d, p=2, dim=-1)
        mf3d = torch.nn.functional.normalize(mf3d, p=2, dim=-1)

        P = self.sinkhorn(mf2d, mf3d, num_points_2d, num_points_3d)

        theta0 = None
        theta = None
        if poseloss:
            # # Skip entire batch if any point-set has < 4 points:
            if (num_points_2d < 4).any() or (num_points_3d < 4).any():
                theta0 = P.new_zeros((P.size(0), 6))
                theta = P.new_zeros((P.size(0), 6))
                return P, theta0, theta
            # # RANSAC:
            p2d_bearings = torch.nn.functional.pad(p2d, (0, 1), "constant", 1.0)
            p2d_bearings = torch.nn.functional.normalize(p2d_bearings, p=2, dim=-1)
            # theta0 = pose
            theta0 = self.min_epropnp(P[:, :-1, :-1], p2d, p3d, num_points_2d, num_points_3d)
            theta = self.solvepnp(P[:, :-1, :-1], p2d_bearings, p3d, theta0)
        return P, theta0, theta


    def inference(self, p2d, p3d, num_points_2d, num_points_3d):
        f3d = p3d
        f2d = p2d
        # Transform f3d to canonical coordinate frame:
        trans = self.stn(f3d.transpose(-2, -1))  # bx3x3
        f3d = torch.bmm(f3d, trans)  # bxnx3
        # Extract features

        f2d = self.FeatureExtractor2d(f2d.transpose(-2, -1))  # 最后加转换就是 b x m x 128
        f3d = self.FeatureExtractor3d(f3d.transpose(-2, -1))  # 不转换是 b x 128 x n

        f3d, f2d = self.gnn(f3d, f2d)

        f3d = self.cov1d(f3d)
        f2d = self.cov1d(f2d)

        mf3d, mf2d = f3d.transpose(-2, -1), f2d.transpose(-2, -1)  # b x m x 128

        # L2 Normalise:
        mf2d = torch.nn.functional.normalize(mf2d, p=2, dim=-1)  # 需要处理128所在维度
        mf3d = torch.nn.functional.normalize(mf3d, p=2, dim=-1)

        P = self.sinkhorn(mf2d, mf3d, num_points_2d, num_points_3d)
        theta = self.min_epropnp(P[:, :-1, :-1], p2d, p3d, num_points_2d, num_points_3d)

        return P, theta  # P[:, :-1, :-1]



