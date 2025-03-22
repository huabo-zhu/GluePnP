import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from utilities.dataset_utilities import MyDataset
from lib.losses import *
import cv2
import os
import time
from model.gluepnp import DeepPnP
from model.epropnp.epropnp import EProPnP6DoF
from model.epropnp.levenberg_marquardt import LMSolver, RSLMSolver

parser = argparse.ArgumentParser(description='PyTorch DeepBlindPnP Test')
parser.add_argument('--dataset', dest='dataset', default='', type=str,
                    help='dataset name')  # no use


def get_dataset(dataset='modelnet40', file_path='I://bpnpnet', eval=True, batch_size=16):
    #  dataset: 'modelnet40'  'megadepth'
    #  file_path: 'I://bpnpnet'
    if eval:
        val_dataset = MyDataset('valid', dataset, file_path, 1, preprocessed=True, sort=0)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=5, drop_last=True,
            collate_fn=None)
        return val_loader
    else:
        train_dataset = MyDataset('train', dataset, file_path, batch_size, 1000, preprocessed=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=10, drop_last=True,
            collate_fn=None)
        return train_loader


def run_method(data_name='modelnet40', data_path='I://bpnpnet', pth_path=''):
    data_loader = get_dataset(data_name, data_path)
    frac_outliers = 0.0
    epnp_test = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=10,
            init_solver=RSLMSolver(
                dof=6,
                num_points=8,
                num_proposals=128,
                num_iter=5)))
    model = DeepPnP(epnp_test)

    loc = 'cuda:{}'.format(0)

    if data_name == 'modelnet40':
        pth_path = r'./weights/modelnet40.pth.tar'
    else:
        pth_path = r'./weights/megadepth.pth.tar'

    checkpoint = torch.load(pth_path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})".format(pth_path, checkpoint['epoch']))
    model = model.cuda()
    model.eval()
    plot_flag = False
    with torch.no_grad():
        rotation_errors0, rotation_errors, rotation_errorsLM = [], [], []
        translation_errors0, translation_errors, translation_errorsLM = [], [], []
        reprojection_errors0, reprojection_errors, reprojection_errorsLM, reprojection_errorsGT = [], [], [], []
        num_inliers0, num_inliers, num_inliersLM, num_inliersGT = [], [], [], []
        num_points_2d_list, num_points_3d_list = [], []
        inference_times = []
        for batch_index, all_data in enumerate(data_loader):
            if batch_index % 10 == 0:
                print(batch_index)
            p2d = all_data["p2d"].float()
            p3d = all_data["p3d"].float()
            R_gt = all_data["R_gt"].float()
            t_gt = all_data["t_gt"].float()
            C_gt = all_data["W_gt_p3d"]
            num_points_2d = all_data["p2d_num"]
            num_points_3d = all_data["p3d_num"]

            if frac_outliers == 0.0:
                m = p2d.size(-2)
                C_gt = torch.nn.functional.one_hot(C_gt, num_classes=(m + 1))[:, :, :m].transpose(-2, -1).float()
            elif frac_outliers > 0.0:
                bb2d_min = p2d.min(dim=-2)[0]
                bb2d_width = p2d.max(dim=-2)[0] - bb2d_min
                bb3d_min = p3d.min(dim=-2)[0]
                bb3d_width = p3d.max(dim=-2)[0] - bb3d_min
                num_outliers = int(frac_outliers * p2d.size(-2))
                p2d_outliers = bb2d_width * torch.rand_like(p2d[:, :num_outliers, :]) + bb2d_min
                p3d_outliers = bb3d_width * torch.rand_like(p3d[:, :num_outliers, :]) + bb3d_min
                p2d = torch.cat((p2d, p2d_outliers), -2)
                p3d = torch.cat((p3d, p3d_outliers), -2)
                num_points_2d = num_points_2d + num_outliers
                num_points_3d = num_points_3d + num_outliers
                # Expand C_gt with outlier indices (b x n index tensor with outliers indexed by m)
                b = p2d.size(0)
                m = p2d.size(-2)
                outlier_indices = C_gt.new_full((b, num_outliers), m)
                C_gt = torch.cat((C_gt, outlier_indices), -1)
                C_gt = torch.nn.functional.one_hot(C_gt, num_classes=(m + 1))[:, :, :m].transpose(-2, -1).float()
                # For memory reasons, if num_points > 10000, downsample first
                if p2d.size(-2) > 10000:
                    idx = torch.randint(p2d.size(-2), size=(10000,))
                    p2d = p2d[:, idx, :]
                    p3d = p3d[:, idx, :]
                    num_points_2d = p2d.size(-2)
                    num_points_3d = p3d.size(-2)
                    C_gt = C_gt[:, idx, :]
                    C_gt = C_gt[:, :, idx]

            p2d = p2d.cuda(0, non_blocking=True)
            p3d = p3d.cuda(0, non_blocking=True)
            R_gt = R_gt.cuda(0, non_blocking=True)
            t_gt = t_gt.cuda(0, non_blocking=True)
            C_gt = C_gt.cuda(0, non_blocking=True)
            num_points_2d = num_points_2d.cuda(0, non_blocking=True)
            num_points_3d = num_points_3d.cuda(0, non_blocking=True)

            start_time = time.time()
            # Compute output

            P, theta0 = model.inference(p2d, p3d, num_points_2d, num_points_3d)

            inference_time = (time.time() - start_time)
            # Compute refined pose estimate:
            # 1. Find inliers based on RANSAC estimate
            inlier_threshold = 1.0 * math.pi / 180.0  # 1 degree threshold for LM
            C = correspondenceMatricesTheta(theta0, p2d, p3d, inlier_threshold)
            K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
            dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
            thetaLM = P.new_zeros((P.size(0), 6))
            inlier_indices = C[0, ...].nonzero(as_tuple=True)  # Assumes test batch size = 1
            # Skip if point-set has < 4 inlier points:
            if (inlier_indices[0].size()[0] >= 4):
                p2d_np = p2d[0, inlier_indices[0], :].cpu().numpy()
                p3d_np = p3d[0, inlier_indices[1], :].cpu().numpy()

                if theta0.shape[1] == 7:
                    R = geo.quaternion_to_matrix(theta0[0, 3:])
                    t = theta0[0, :3]
                    rvec = R.cpu().numpy()
                    rvec, _ = cv2.Rodrigues(rvec)
                    tvec = t.cpu().numpy()
                else:
                    rvec = theta0[0, :3].cpu().numpy()
                    tvec = theta0[0, 3:].cpu().numpy()

                rvec, tvec = cv2.solvePnPRefineLM(p3d_np, p2d_np, K, dist_coeff, rvec, tvec)
                if rvec is not None and tvec is not None:
                    thetaLM[0, :3] = torch.as_tensor(rvec, dtype=P.dtype, device=P.device).squeeze(-1)
                    thetaLM[0, 3:] = torch.as_tensor(tvec, dtype=P.dtype, device=P.device).squeeze(-1)

            inlier_threshold = 0.1 * math.pi / 180.0  # 0.1 degree threshold for reported inlier count
            rotation_errors0 += [rotationErrorsTheta(theta0, R_gt, eps=0.0).item()]
            rotation_errorsLM += [rotationErrorsTheta(thetaLM, R_gt, eps=0.0).item()]
            translation_errors0 += [translationErrorsTheta(theta0, t_gt).item()]
            translation_errorsLM += [translationErrorsTheta(thetaLM, t_gt).item()]
            reprojection_errors0 += [reprojectionErrorsTheta(theta0, p2d, p3d, C_gt, eps=0.0).item()]
            reprojection_errorsLM += [reprojectionErrorsTheta(thetaLM, p2d, p3d, C_gt, eps=0.0).item()]
            reprojection_errorsGT += [reprojectionErrors(R_gt, t_gt, p2d, p3d, C_gt, eps=0.0).item()]
            num_inliers0 += [numInliersTheta(theta0, p2d, p3d, inlier_threshold).item()]
            num_inliersLM += [numInliersTheta(thetaLM, p2d, p3d, inlier_threshold).item()]
            num_inliersGT += [numInliers(R_gt, t_gt, p2d, p3d, inlier_threshold).item()]
            num_points_2d_list += [num_points_2d[0].item()]
            num_points_3d_list += [num_points_3d[0].item()]
            inference_times += [inference_time]

        print("rotation error0:{}".format(sum(rotation_errors0) / len(rotation_errors0) * 57.2958))
        print("translation error0:{}".format(sum(translation_errors0) / len(translation_errors0)))
        print("reprojection error0:{}".format(sum(reprojection_errors0) / len(reprojection_errors0) * 57.2958))
        print('average time: {}'.format(sum(inference_times) / len(inference_times)))
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./results/' + data_name, exist_ok=True)
        with open('./results/' + data_name + '/results.txt',
                  'w') as save_file:
            save_file.write(
                "rotation_errors0, translation_errors0, reprojection_errors0, num_inliers0, inference_time\n")
            for i in range(len(rotation_errors0)):
                save_file.write(
                    "{:.9f}, {:.9f}, {:.9f}, {},  {:.9f}\n".format(
                        rotation_errors0[i],
                        translation_errors0[i],
                        reprojection_errors0[i],
                        num_inliers0[i],
                        inference_times[i]
                    ))


if __name__ == "__main__":
    run_method(data_name='megadepth')  # dataset: 'modelnet40'  'megadepth'
