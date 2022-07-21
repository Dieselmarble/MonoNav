import os
import time

import cv2
import random
import numpy as np
from PIL import Image
from distutils.version import LooseVersion
from sacred import Experiment
from easydict import EasyDict as edict
import torch
import torchvision.transforms as tf
from models.baseline_same import Baseline as UNet
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import get_coordinate_map
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss
import math
from cv2 import aruco

ex = Experiment()

@ex.main
def predict(_run, _log):
    cfg = edict(_run.config)
    cfg.resume_dir = "pretrained.pt"
    cfg.image_path = "results/capture.png"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build network
    network = UNet(cfg.model)
    if not (cfg.resume_dir == 'None'):
        model_dict = torch.load(cfg.resume_dir, map_location=lambda storage, loc: storage)
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)
    network.eval()

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    bin_mean_shift = Bin_Mean_Shift(device=device)
    k_inv_dot_xy1 = get_coordinate_map(device)
    instance_parameter_loss = InstanceParameterLoss(k_inv_dot_xy1)

    h, w = 192, 256

    with torch.no_grad():
        image = cv2.imread(cfg.image_path)
        # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
        image = cv2.resize(image, (w, h))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('results/resize.png', image)
        image = Image.fromarray(image)
        image = transforms(image)
        image = image.to(device).unsqueeze(0)
        # forward pass
        logit, embedding, _, _, param = network(image)

        prob = torch.sigmoid(logit[0])
        # infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
        _, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, torch.ones_like(logit))

        # fast mean shift
        segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(
            prob, embedding[0], param, mask_threshold=0.1)

        # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned,
        # we thus use avg_pool_2d to smooth the segmentation results
        b = segmentation.t().view(1, -1, h, w)
        pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
        b = pooling_b.view(-1, h * w).t()
        segmentation = b

        # infer instance depth
        instance_loss, instance_depth, instance_abs_disntace, instance_parameter = instance_parameter_loss(
            segmentation, sampled_segmentation, sample_param, torch.ones_like(logit), torch.ones_like(logit), False)

        # return cluster results
        predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)
        # pdb.set_trace()
        # mask out non planar region
        predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
        predict_segmentation = predict_segmentation.reshape(h, w)

        # visualization and evaluation
        image = tensor_to_image(image.cpu()[0])
        mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
        depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
        per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

        # use per pixel depth for non planar region
        depth = depth * (predict_segmentation != 20) + per_pixel_depth * (predict_segmentation == 20)
        np.save(os.path.join('results/', 'depth.npy'), depth)
        # change non planar to zero, so non planar region use the black color
        predict_segmentation += 1
        predict_segmentation[predict_segmentation == 21] = 0
        np.save('results/segmentation.npy', predict_segmentation)
        pred_seg = cv2.resize(np.stack([colors[predict_segmentation, 0],
                                        colors[predict_segmentation, 1],
                                        colors[predict_segmentation, 2]], axis=2), (w, h))
        # blend image
        blend_pred = (pred_seg * 0.7 + image * 0.3).astype(np.uint8)
        mask = cv2.resize((mask * 255).astype(np.uint8), (w, h))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # visualize depth map as PlaneNet
        depth = 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
        depth = cv2.cvtColor(cv2.resize(depth, (w, h)), cv2.COLOR_GRAY2BGR)
        image = np.concatenate((image, pred_seg, blend_pred, mask, depth), axis=1)
        cv2.imwrite('results/result.png', image)


def arucoAnimation(frame, rvec, tvec, ids, corners):
    font = cv2.FONT_HERSHEY_SIMPLEX
    dist = np.array([[0.1563097, -0.50631796, -0.00239552, -0.00120906, 0.4788988]])
    # camera intrinsics
    mtx = np.array([[889.69615837, 0. , 647.5929987],
                    [0.,889.09279472, 351.73822108],
                    [0., 0.,  1.]])

    # 在画面上 标注auruco标签的各轴
    for i in range(rvec.shape[0]):
        aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
        aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    ###### 距离估计 #####
    distance = tvec[0][0][2]  # 单位是米
    # 显示距离
    cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), font, 0.75, (0, 255, 0), 2,
                cv2.LINE_AA)
    ###### 角度估计 #####
    # 考虑Z轴（蓝色）的角度
    # 本来正确的计算方式如下，但是由于蜜汁相机标定的问题，实测偏航角度能最大达到104°所以现在×90/104这个系数作为最终角度
    deg = rvec[0][0][2] / math.pi * 180
    # deg=rvec[0][0][2]/math.pi*180*90/104
    # 旋转矩阵到欧拉角
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec[0], R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:  # 偏航，俯仰，滚动
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # 偏航，俯仰，滚动换成角度
    rx = x * 180.0 / math.pi
    ry = y * 180.0 / math.pi
    rz = z * 180.0 / math.pi
    cv2.putText(frame, 'deg_z:' + str(ry)[:4] + str('deg'), (0, 140), font, 0.75, (0, 255, 0), 2,
                cv2.LINE_AA)
    # print("偏航，俯仰，滚动",rx,ry,rz)

def getPerspectiveTransform():
    font = cv2.FONT_HERSHEY_SIMPLEX
    dist = np.array([[0.1563097, -0.50631796, -0.00239552, -0.00120906, 0.4788988]])
    # camera intrinsics
    mtx = np.array([[889.69615837, 0. , 647.5929987],
                    [0.,889.09279472, 351.73822108],
                    [0., 0.,  1.]])

    video = cv2.VideoCapture(1)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while (True):
        _, frame = video.read()
        corners, ids, _ = aruco.detectMarkers(frame, aruco.Dictionary_get(aruco.DICT_6X6_1000))
        if ids is not None:
            for id_ in range(len(ids)):
                id = ids[id_][0]
                # frame is '/world'
                if id == 1:
                    dts = np.float32([[640, 500], [640, 530], [670, 500], [670, 530]])
                    pts = corners[id_][0]
                    norms = [np.linalg.norm(p) for p in pts]
                    rank = np.argsort(norms)  # min to max
                    pts_new = pts[rank]
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.09, mtx, dist)
                    (rvec - tvec).any()  # get rid of that nasty numpy value array error
                    arucoAnimation(frame, rvec, tvec, ids, corners)
                    # record transform matrix
                    M = cv2.getPerspectiveTransform(pts_new, dts)
                    img = frame
                    result = cv2.warpPerspective(frame, M, (1280, 720))
                    cv2.imshow('transformed', result)
                    time.sleep(0.25)
                else:
                    pass
        else:
            # DRAW "NO IDS"
            cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.namedWindow('Map')
        cv2.resizeWindow('Map', 1280, 720)
        cv2.imshow("Map", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.imwrite(os.path.join('results/', 'capture.png'), img) # save original map
            print("capture.png successfuly!")
            print("-------------------------")
            break
    cv2.destroyAllWindows()
    return img, M

if __name__ == '__main__':
    # get transformed BEV segmentation
    img, Hr = getPerspectiveTransform()
    cv2.destroyAllWindows()  # 释放并销毁窗口
    # image segmentation and depth estimation
    #assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
    #    'PyTorch>=0.4.0 is required'
    ex.add_config('./configs/predict.yaml')
    ex.run_commandline()
    np.save('results/perspective.npy', Hr)
