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

from model.cnnmodel import DBPnP
# from model.model_hungarian import DBPnP  # At test time only

parser = argparse.ArgumentParser(description='PyTorch DeepBlindPnP Test')
parser.add_argument('--dataset', dest='dataset', default='', type=str,
                    help='dataset name')  # no use


def get_dataset(dataset='megadepth', file_path='I://bpnpnet', batch_size=16):
    #  dataset: 'modelnet40' or 'megadepth'
    #  file_path: 'I://bpnpnet'
    val_dataset = MyDataset('valid', dataset, file_path, 1, preprocessed=True, sort=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=5, drop_last=True,
        collate_fn=None)
    return val_loader

def eval_method(data_name='megadepth', data_path='I://bpnpnet', eval_data=None, method='r1ppnp', ):
    data_loader = get_dataset(data_name, data_path)
    # 分直接model输出的，还是从文件读取的位姿
    deep_method_list = ['ours', 'dbpnp',]
    matlab_method_list = ['r1ppnp', 'vpnp']
    class_method_list = ['ransac']
    frac_outliers = 0.0
    if method in matlab_method_list:
        # r和t的mat文件
        r_path = eval_data + '//r_est.mat'
        t_path = eval_data + '//t_est.mat'
        est_r = loadmat(r_path)
        est_r = est_r['r_est']
        est_t = loadmat(t_path)
        est_t = est_t['t_est']

        # b = a['R_gt_0']
        rotation_errors, translation_errors, reprojection_errors = [], [], []
        num_inliers = []

        for batch_index, all_data in enumerate(data_loader):
            # (p2d, p3d, R_gt, t_gt, C_gt, num_points_2d, num_points_3d)
            p2d = all_data["p2d"]
            p3d = all_data["p3d"]
            R_gt = all_data["R_gt"]
            t_gt = all_data["t_gt"]
            W_gt_p3d = all_data["W_gt_p3d"]
            # p2d_num = all_data["num_points_2d"][batch_index]
            # p3d_num = all_data["num_points_3d"][batch_index]

            m = p2d.size(-2)
            W_gt = torch.nn.functional.one_hot(W_gt_p3d, num_classes=(m + 1))[:, :, :m].transpose(-2, -1).float()
            R_est_i = est_r['R_gt_{}'.format(batch_index)][0, 0]
            t_est_i = est_t['t_gt_{}'.format(batch_index)][0, 0]
            R_est_i, _ = cv2.Rodrigues(R_est_i)

            theta = np.zeros([1, 6], dtype=np.float32)
            theta[:, :3] = R_est_i.reshape(1, 3)
            theta[:, 3:] = t_est_i.reshape(1, 3)
            theta = torch.from_numpy(theta)
            rotation_errors += [rotationErrorsTheta(theta, R_gt, eps=0.0).item()]
            translation_errors += [translationErrorsTheta(theta, t_gt).item()]
            reprojection_errors += [reprojectionErrorsTheta(theta, p2d, p3d, W_gt, eps=0.0).item()]
            inlier_threshold = 0.1 * math.pi / 180.0
            num_inliers += [numInliersTheta(theta, p2d, p3d, inlier_threshold).item()]

        os.makedirs('./results', exist_ok=True)
        os.makedirs('./results/' + data_name, exist_ok=True)
        os.makedirs('./results/' + data_name + '/' + method, exist_ok=True)
        with open('./results/' + data_name + '/' + method + '/results.txt', 'w') as save_file:
            save_file.write("rotation_errors, translation_errors, reprojection_errors, num_inliers")
            for i in range(len(rotation_errors)):
                save_file.write(
                    "{:.9f}, {:.9f}, {:.9f}\n".format(rotation_errors[i], translation_errors[i],
                                                      reprojection_errors[i], num_inliers[i]))

    elif method in class_method_list:
        rotation_errors, translation_errors, reprojection_errors = [], [], []
        num_inliers, inference_times = [], []

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


            # solvepnpransac
            # 1. Choose top k correspondences:

            K = np.float32(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
            dist_coeff = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
            k = 5000
            # 2. Loop over batch and run RANSAC:
            num_points_ransac = min(k, round(2.0 * num_points_2d[0].float().item()),
                                    round(2.0 * num_points_3d[0].float().item()))
            num_points_ransac = min(k, max(num_points_ransac, 10))  # At least 10 points
            p2d_np = p2d[0, :num_points_ransac, :].cpu().numpy()
            p3d_np = p3d[0, :num_points_ransac, :].cpu().numpy()
            start_time = time.time()
            retval, R_est_i, t_est_i, inliers = cv2.solvePnPRansac(
                p3d_np, p2d_np, K, dist_coeff,
                iterationsCount=1000,
                reprojectionError=0.01,
                confidence=0.99,
                flags=cv2.SOLVEPNP_P3P)
            inference_time = (time.time() - start_time)
            if not retval:
                R_est_i = np.zeros((1, 3))
                t_est_i = np.zeros((1, 3))

            theta = np.zeros([1, 6], dtype=np.float32)
            theta[:, :3] = R_est_i.reshape(1, 3)
            theta[:, 3:] = t_est_i.reshape(1, 3)
            theta = torch.from_numpy(theta)
            rotation_errors_temp = rotationErrorsTheta(theta, R_gt, eps=0.0).item()
            translation_errors_temp = translationErrorsTheta(theta, t_gt).item()
            reprojection_errors_temp = reprojectionErrorsTheta(theta, p2d, p3d, C_gt, eps=0.0).item()
            rotation_errors += [rotation_errors_temp]
            translation_errors += [translation_errors_temp]
            reprojection_errors += [reprojection_errors_temp]

            inlier_threshold = 0.1 * math.pi / 180.0
            num_inliers += [numInliersTheta(theta, p2d, p3d, inlier_threshold).item()]
            inference_times += [inference_time]
            if batch_index % 10 == 0:
                print(
                    'Index: {},Rotation Error: {:.4f}, Translation Error: {:.4f}, Reprojection Error: {:.4f}'
                    .format(batch_index, rotation_errors_temp * 57.2958, translation_errors_temp,
                            reprojection_errors_temp * 57.2958))

        print("average rotation error0:{}".format(sum(rotation_errors) / len(rotation_errors) * 57.2958))
        print("average translation error0:{}".format(sum(translation_errors) / len(translation_errors)))
        print("average reprojection error0:{}".format(sum(reprojection_errors) / len(reprojection_errors) * 57.2958))
        print('average time: {}'.format(sum(inference_times) / len(inference_times)))

        rotation_percentage = np.percentile(rotation_errors, [25, 50, 75])
        print('Rotation Error: 25th percentile: {:.4f}, 50th percentile: {:.4f}, 75th percentile: {:.4f}'.
              format(rotation_percentage[0] * 57.2958, rotation_percentage[1] * 57.2958,
                     rotation_percentage[2] * 57.2958))
        translation_percentage = np.percentile(translation_errors, [25, 50, 75])
        print('Translation Error: 25th percentile: {:.4f}, 50th percentile: {:.4f}, 75th percentile: {:.4f}'.
              format(translation_percentage[0], translation_percentage[1], translation_percentage[2]))
        reprojection_percentage = np.percentile(reprojection_errors, [25, 50, 75])
        print('Reprojection Error: 25th percentile: {:.4f}, 50th percentile: {:.4f}, 75th percentile: {:.4f}'.
              format(reprojection_percentage[0] * 57.2958, reprojection_percentage[1] * 57.2958,
                     reprojection_percentage[2] * 57.2958))

        os.makedirs('./results', exist_ok=True)
        os.makedirs('./results/' + data_name, exist_ok=True)
        os.makedirs('./results/' + data_name + '/' + method, exist_ok=True)
        with open('./results/' + data_name + '/' + method + '/results.txt', 'w') as save_file:
            save_file.write(
                "rotation_errors, translation_errors, reprojection_errors, num_inliers, inference_time\n")
            for i in range(len(rotation_errors)):
                save_file.write(
                    "{:.9f}, {:.9f}, {:.9f}, {}, {:.9f}\n".format(rotation_errors[i], translation_errors[i],
                                                      reprojection_errors[i], num_inliers[i], inference_times[i]))
        print("total: {}".format((batch_index + 1)))

    elif method in deep_method_list:
        if method == 'ours':
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
        else:
            args = parser.parse_args()
            model = DBPnP(args)

        loc = 'cuda:{}'.format(0)
        if method == 'ours':
            if data_name == 'modelnet40':
                pth_path = r'./weights/modelnet40.pth.tar'
            else:
                pth_path = r'./weights/megadepth.pth.tar'
        else:
            if data_name == 'modelnet40':
                pth_path = r'./weights/dbpnp_modelnet40.pth.tar'
            else:
                pth_path = r'./weights/dbpnp_megadepth.pth.tar'

        checkpoint = torch.load(pth_path, map_location=loc)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(pth_path, checkpoint['epoch']))
        model = model.cuda()
        model.eval()
        rotation_errors0, rotation_errors = [], []
        translation_errors0, translation_errors= [], []
        reprojection_errors0, reprojection_errors, reprojection_errorsGT = [], [], []
        num_inliers0, num_inliers, num_inliersLM, num_inliersGT = [], [], [], []
        num_points_2d_list, num_points_3d_list = [], []
        correspondence = []
        inference_times = []
        with torch.no_grad():
            for batch_index, all_data in enumerate(data_loader):
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
                if method == 'ours':
                    P, theta0 = model.inference(p2d, p3d, num_points_2d, num_points_3d)
                else:
                    P, theta0, theta = model(p2d, p3d, num_points_2d, num_points_3d, poseloss=10)

                inference_time = (time.time() - start_time)

                inlier_threshold = 0.1 * math.pi / 180.0  # 0.1 degree threshold for reported inlier count
                rotation_errors_temp = rotationErrorsTheta(theta0, R_gt, eps=0.0).item()
                translation_errors_temp = translationErrorsTheta(theta0, t_gt).item()
                reprojection_errors_temp = reprojectionErrorsTheta(theta0, p2d, p3d, C_gt, eps=0.0).item()
                if method == 'ours':
                    correspondence_temp = correspondenceLoss(P[:, :-1, :-1], C_gt).item()
                else:
                    correspondence_temp = correspondenceLoss(P, C_gt).item()
                rotation_errors0 += [rotation_errors_temp]
                translation_errors0 += [translation_errors_temp]
                reprojection_errors0 += [reprojection_errors_temp]
                reprojection_errorsGT += [reprojectionErrors(R_gt, t_gt, p2d, p3d, C_gt, eps=0.0).item()]
                num_inliers0 += [numInliersTheta(theta0, p2d, p3d, inlier_threshold).item()]
                num_inliersGT += [numInliers(R_gt, t_gt, p2d, p3d, inlier_threshold).item()]
                num_points_2d_list += [num_points_2d[0].item()]
                num_points_3d_list += [num_points_3d[0].item()]
                inference_times += [inference_time]
                correspondence += [correspondence_temp]
                if batch_index % 10 == 0:
                    print(
                        'Index: {},correspondence Error: {:.4f}, Rotation Error: {:.4f}, Translation Error: {:.4f}, '
                        'Reprojection Error: {:.4f}'
                        .format(batch_index, rotation_errors_temp, rotation_errors_temp * 57.2958, translation_errors_temp,
                                reprojection_errors_temp * 57.2958))

                if method != 'ours':
                    rotation_errors += [rotationErrorsTheta(theta, R_gt, eps=0.0).item()]
                    translation_errors += [translationErrorsTheta(theta, t_gt).item()]
                    reprojection_errors += [reprojectionErrorsTheta(theta, p2d, p3d, C_gt, eps=0.0).item()]
                    num_inliers += [numInliersTheta(theta, p2d, p3d, inlier_threshold).item()]

            if method != 'ours':
                print("average rotation error:{}".format(sum(rotation_errors) / len(rotation_errors) * 57.2958))
                print("average translation error:{}".format(sum(translation_errors) / len(translation_errors)))
                print("average reprojection error:{}".format(sum(reprojection_errors) / len(reprojection_errors) * 57.2958))
            print("average rotation error0:{}".format(sum(rotation_errors0) / len(rotation_errors0) * 57.2958))
            print("average translation error0:{}".format(sum(translation_errors0) / len(translation_errors0)))
            print("average reprojection error0:{}".format(sum(reprojection_errors0) / len(reprojection_errors0) * 57.2958))
            print("average correspondence error:{}".format(sum(correspondence) / len(correspondence)))
            print('average time: {}'.format(sum(inference_times) / len(inference_times)))

            rotation_percentage = np.percentile(rotation_errors0, [25, 50, 75])
            print('Rotation Error: 25th percentile: {:.4f}, 50th percentile: {:.4f}, 75th percentile: {:.4f}'.
                  format(rotation_percentage[0] * 57.2958, rotation_percentage[1] * 57.2958,
                         rotation_percentage[2] * 57.2958))
            translation_percentage = np.percentile(translation_errors0, [25, 50, 75])
            print('Translation Error: 25th percentile: {:.4f}, 50th percentile: {:.4f}, 75th percentile: {:.4f}'.
                  format(translation_percentage[0], translation_percentage[1], translation_percentage[2]))
            reprojection_percentage = np.percentile(reprojection_errors0, [25, 50, 75])
            print('Reprojection Error: 25th percentile: {:.4f}, 50th percentile: {:.4f}, 75th percentile: {:.4f}'.
                  format(reprojection_percentage[0] * 57.2958, reprojection_percentage[1] * 57.2958,
                         reprojection_percentage[2] * 57.2958))

            os.makedirs('./results', exist_ok=True)
            os.makedirs('./results/' + data_name, exist_ok=True)
            os.makedirs('./results/' + data_name + '/' + method, exist_ok=True)
            with open('./results/' + data_name + '/' + method + '/results.txt',
                      'w') as save_file:
                if method == 'ours':
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
                elif method == 'dbpnp':
                    save_file.write(
                        "rotation_errors0, rotation_errors, translation_errors0, translation_errors, "
                        "reprojection_errors0, reprojection_errors, "
                        "reprojection_errorsGT, num_inliers0, num_inliers, num_inliersLM, num_inliersGT, num_points_2d, "
                        "num_points_3d, inference_time\n")
                    for i in range(len(rotation_errors0)):
                        save_file.write(
                            "{:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {}, {}, "
                            "{}, {}, {}, {}, {:.9f}\n".format(
                                rotation_errors0[i], rotation_errors[i],
                                translation_errors0[i], translation_errors[i],
                                reprojection_errors0[i], reprojection_errors[i],
                                reprojection_errorsGT[i],
                                num_inliers0[i], num_inliers[i], num_inliersLM[i], num_inliersGT[i],
                                num_points_2d_list[i], num_points_3d_list[i],
                                inference_times[i]
                            ))


    else:
        print('method error')
        return


if __name__ == "__main__":
    eval_method(data_name='megadepth', method='ours')  # eval megadepth dataset
    # eval_method(data_name='modelnet40', method='ours')  #  eval modelnet40 dataset
    # method=ours dbpnp ransac (r1ppnp vpnp)->matlab output
