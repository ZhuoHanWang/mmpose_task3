import numpy as np
import math
import json
from .coco import COCO
import skimage.io as io
import os
import matplotlib.pyplot as plt
from .autoencoder import Autoencoder
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .spine import dataset_info

BASE_DIR = '/pangyan/wzh/mmpose_task/task4'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'spine')
IMG_PREFIX = os.path.join(DATA_DIR, 'images')
def get_angle(keypoints):
    # 计算每个点连接后的vector

    kp_vec = []
    for i in range(len(keypoints)-1):
        x1 = keypoints[i][0]
        y1 = keypoints[i][1]
        x2 = keypoints[i + 1][0]
        y2 = keypoints[i + 1][1]
        kp_vec.append(np.array([x2 - x1, y2 - y1]))

    kp_vec.append(kp_vec[-1])  # 最后一个点和倒数第二个点视为同一个

    print(kp_vec)
    # 计算垂直向量
    kp_vec = [np.array([-vec[1], vec[0]]) for vec in kp_vec]
    print(kp_vec)
    # 计算每个点的 tanθ (斜率) #垂直向量的斜率
    tan_theta = [vec[1] / vec[0] for vec in kp_vec]
    print(tan_theta)

    radian_value = map(math.atan, tan_theta)  # 正切转弧度值
    theta = list(map(math.degrees, radian_value))  # 弧度值转角度值

    # Angle matrix 夹角矩阵
    angle_matrix = np.zeros([len(keypoints), len(keypoints)])
    for i in range(len(keypoints)):
        for j in range(len(keypoints)):
            angle_matrix[i, j] = abs(theta[i] - theta[j])


    return tan_theta, theta, angle_matrix

def cobb_caculate(kp_list):

    keypoints = []
    for i in range(17):
        x = kp_list[i * 3]
        y = kp_list[i * 3 + 1]
        keypoint = np.array([x, y])
        keypoints.append(keypoint)

    tan_theta, theta, angle_matrix = get_angle(keypoints)

    # 找到最大值及其索引
    slope_max = np.max(tan_theta)
    slope_min = np.min(tan_theta)

    max_index = np.argmax(tan_theta)
    min_index = np.argmin(tan_theta)

    slope_2_index = min(max_index, min_index)
    slope_2 = tan_theta[slope_2_index]

    slope_3_index = max(max_index, min_index)
    slope_3 = tan_theta[slope_3_index]

    MT_angle_radians = math.atan(abs((slope_2 - slope_3) / (1 + slope_2 * slope_3)))
    MT = (180 / math.pi) * MT_angle_radians

    if slope_2_index==0:
        slope_1 = slope_2
        slope_1_index = slope_2_index
        PT = 0
    else:
        PT_keypoints = keypoints[:slope_2_index+1]
        PT_tan_theta, PT_theta, PT_angle_matrix = get_angle(PT_keypoints)
        if tan_theta[slope_2_index] == slope_max:
            slope_1 = np.min(PT_tan_theta)
            slope_1_index = np.argmin(PT_tan_theta)
        elif tan_theta[slope_2_index] == slope_min:
            slope_1 = np.max(PT_tan_theta)
            slope_1_index = np.argmax(PT_tan_theta)
        else:
            assert 0
        PT_angle_radians = math.atan(abs((slope_1 - slope_2) / (1 + slope_1 * slope_2)))
        PT = (180 / math.pi) * PT_angle_radians


    if slope_3_index==16:
        slope_4 = slope_3
        slope_4_index = slope_3_index
        TL = 0
    else:
        down = max(max_index, min_index)
        TL_keypoionts = keypoints[down:]
        TL_tan_theta, TL_theta, TL_angle_matrix = get_angle(TL_keypoionts)
        if tan_theta[slope_3_index] == slope_max:
            slope_4 = np.min(TL_tan_theta)
            slope_4_index = np.argmin(TL_tan_theta)
        elif tan_theta[slope_3_index] == slope_min:
            slope_4 = np.max(TL_tan_theta)
            slope_4_index = np.argmax(TL_tan_theta)
        else:
            assert 0
        TL_angle_radians = math.atan(abs((slope_3 - slope_4) / (1 + slope_3 * slope_4)))
        TL = (180 / math.pi) * TL_angle_radians




    slope_list = [slope_1, slope_2, slope_3, slope_4]
    slope_index_list = [slope_1_index, slope_2_index, slope_3_index, slope_4_index]

    return slope_list, slope_index_list, tan_theta, theta, PT, MT, TL


# Calculate the cobb angle after masking the anomaly
def cobb_correct(kp_list,  res_idx, anomaly_matrix):
    keypoints = []
    for i in range(17):
        x = kp_list[i * 3]
        y = kp_list[i * 3 + 1]
        keypoint = np.array([x, y])
        keypoints.append(keypoint)

    tan_theta, theta, angle_matrix = get_angle(keypoints)

    print("Uncorrected:", tan_theta)

    anomaly_vector = anomaly_matrix[res_idx]
    mask_tan_theta = [0 if m else value for m, value in zip(anomaly_vector, tan_theta)]
    mask_theta = [0 if m else value for m, value in zip(anomaly_vector, theta)]

    print("Corrected:", mask_tan_theta)

    # 找到最大值、最小值及其索引
    slope_max = np.max(mask_tan_theta)
    slope_min = np.min(mask_tan_theta)

    print("Max Slope:", slope_max)
    print("Min Slope:", slope_min)

    max_index = np.argmax(mask_tan_theta)
    min_index = np.argmin(mask_tan_theta)

    print("Max Index:", max_index)
    print("Min Index:", min_index)

    slope_2_index = min(max_index, min_index)
    slope_2 = mask_tan_theta[slope_2_index]

    print("Slope 2:", slope_2)
    print("Slope 2 Index:", slope_2_index)

    slope_3_index = max(max_index, min_index)
    slope_3 = mask_tan_theta[slope_3_index]

    print("Slope 3:", slope_3)
    print("Slope 3 Index:", slope_3_index)

    MT_angle_radians = math.atan(abs((slope_2 - slope_3) / (1 + slope_2 * slope_3)))
    MT = (180 / math.pi) * MT_angle_radians

    if slope_2_index==0:
        slope_1 = slope_2
        slope_1_index = slope_2_index
        PT = 0
    else:

        mask_PT_tan_theta = mask_tan_theta[:slope_2_index + 1]

        print("Corrected PT Theta:", mask_PT_tan_theta)

        if mask_tan_theta[slope_2_index] == slope_max:
            slope_1 = np.min(mask_PT_tan_theta)
            slope_1_index = np.argmin(mask_PT_tan_theta)
        elif mask_tan_theta[slope_2_index] == slope_min:
            slope_1 = np.max(mask_PT_tan_theta)
            slope_1_index = np.argmax(mask_PT_tan_theta)
        else:
            assert 0

        PT_angle_radians = math.atan(abs((slope_1 - slope_2) / (1 + slope_1 * slope_2)))
        PT = (180 / math.pi) * PT_angle_radians

    print("Slope 1:", slope_1)
    print("Slope 1 index:",slope_1_index)

    if slope_3_index==16:
        slope_4 = slope_3
        slope_4_index = slope_3_index
        TL = 0
    else:

        mask_TL_tan_theta = mask_tan_theta[slope_3_index:]

        print("Corrected TL Theta:", mask_TL_tan_theta)

        if tan_theta[slope_3_index] == slope_max:
            slope_4 = np.min(mask_TL_tan_theta)
            slope_4_index = np.argmin(mask_TL_tan_theta) + slope_3_index
        elif tan_theta[slope_3_index] == slope_min:
            slope_4 = np.max(mask_TL_tan_theta)
            slope_4_index = np.argmax(mask_TL_tan_theta) + slope_3_index
        else:
            assert 0

        TL_angle_radians = math.atan(abs((slope_3 - slope_4) / (1 + slope_3 * slope_4)))
        TL = (180 / math.pi) * TL_angle_radians

    print("Slope 4:", slope_1)
    print("Slope 4 index:",slope_1_index)

    slope_list = [slope_1, slope_2, slope_3, slope_4]
    slope_index_list = [slope_1_index, slope_2_index, slope_3_index, slope_4_index]


    return slope_list, slope_index_list, mask_tan_theta, mask_theta, PT, MT, TL


def calculate_cobb_mae(true_angles, pred_angles):


    true_angles = np.array(true_angles)
    pred_angles = np.array(pred_angles)
    angle_num = true_angles.shape[1]

    all_angle_mae = []
    for i in range(angle_num):


        # 确保输入为NumPy数组
        true = true_angles[:, i]
        pred = pred_angles[:, i]

        # 计算最小角度差
        diff = np.abs(true - pred)
        min_diff = np.minimum(diff, 180 - diff)

        # 计算MAE
        mae = np.mean(min_diff)
        all_angle_mae.append(mae)
    return all_angle_mae

def symmetric_mean_absolute_percentage(y_true, y_pred):
    """
    计算对称平均绝对百分比误差 (SMAPE)

    :param y_true: 实际值 (numpy 数组或列表)
    :param y_pred: 预测值 (numpy 数组或列表)
    :return: SMAPE 值
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 避免除以零
    denominator = np.abs(y_true) + np.abs(y_pred)


    diff = np.sum(np.abs(y_true - y_pred), axis=1) / np.sum(denominator, axis=1)

    # # 将无穷大值替换为 0
    # diff[denominator == 0] = 0.0

    return 100 * np.mean(diff)




def circular_mean_absolute_error(gt_angles, pred_angles):
    # 计算角度差
    deltas = np.abs(np.array(pred_angles) - np.array(gt_angles))

    # 计算角度差的余弦和正弦
    cos_deltas = np.cos(deltas)
    sin_deltas = np.sin(deltas)


    # 计算每个样本的方向误差
    rads = np.arctan2(np.mean(sin_deltas,axis=1), np.mean(cos_deltas,axis=1))

    # 转角度值并计算平均值
    cmae = np.degrees(np.mean(rads))

    return cmae

def show_cobb(results, annFile, detector_path, save_path, img_prefix=IMG_PREFIX):

    cobb_visual_path = os.path.join(save_path, 'cobb_visual')

    if not os.path.exists(cobb_visual_path):
        os.makedirs(cobb_visual_path)


    with open(annFile, 'r', encoding='utf-8') as file:
        ann = json.load(file)

    skeleton = ann['categories'][0]['skeleton']

    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)


    #gt——cobb angle and pred——cobb angle collection
    cobb_true = []
    cobb_uncorrected_dict = {}
    uncorrected_tan_theta = []
    cobb_uncorrected = []
    for res_idx, result in enumerate(results):
        img_name = result['image_paths'][0].split('/')[-1]

        for img in imgs:

            if img['file_name'] == img_name:
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)

                gt_kp_list = anns[0]['keypoints']
                gt_slope_list, gt_slope_index_list, gt_tan_theta, gt_theta, gt_PT, gt_MT, gt_TL = cobb_caculate(gt_kp_list)

                cobb_true.append([gt_PT, gt_MT, gt_TL])


        pred_kp_list = np.array(result['preds'][0]).reshape(-1)

        pred_slope_list, pred_slope_index_list, pred_tan_theta, pred_theta, pred_PT, pred_MT, pred_TL = cobb_caculate(pred_kp_list)

        pred_PT_str = str(pred_PT) + '°'
        pred_MT_str = str(pred_MT) + '°'
        pred_TL_str = str(pred_TL) + '°'
        cobb_uncorrected_dict[img_name + '-PT'] = pred_PT_str
        cobb_uncorrected_dict[img_name + '-MT'] = pred_MT_str
        cobb_uncorrected_dict[img_name + '-TL'] = pred_TL_str
        cobb_uncorrected.append([pred_PT, pred_MT, pred_TL])

        uncorrected_tan_theta.append(pred_tan_theta)

    #anomalous slope detection
    anomaly_matrix = []
    for i in range(17):
        # 2. 数据预处理
        tan_theta_item = np.array(uncorrected_tan_theta)[:, i]  # 获取指定列

        tan_theta_item = np.expand_dims(tan_theta_item, axis=-1)

        scaler = StandardScaler()
        tan_theta_item= scaler.fit_transform(tan_theta_item)  # 标准化数据

        tan_theta_item = torch.FloatTensor(tan_theta_item)

        autoencoder = Autoencoder()
        autoencoder.load_state_dict(torch.load(os.path.join(detector_path , str(i) + '.pth')))

        autoencoder.eval()
        with torch.no_grad():
            reconstructed = autoencoder(tan_theta_item)
            reconstruction_error = torch.mean((reconstructed - tan_theta_item) ** 2, dim=1).numpy()

        # 6. 设置阈值并进行异常检测
        threshold = np.percentile(reconstruction_error, 70)  # 设定阈值为重构误差的70百分位数
        anomalies = reconstruction_error > threshold
        anomaly_matrix.append(anomalies)
    anomaly_matrix = list(zip(*anomaly_matrix))


    #cobb angle correction
    cobb_corrected = []
    cobb_corrected_dict = {}
    for res_idx, result in enumerate(results):

        img_name = result['image_paths'][0].split('/')[-1]
        I = io.imread(os.path.join(img_prefix, result['image_paths'][0].split('/')[-1]))

        plt.imshow(I)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        plt.axis('off')

        if len(result['preds']) == 0:
            print('Output error!')
            assert 0

        x, y, v = [], [], []
        for pred in result['preds'][0]:
            x.append(pred[0])
            y.append(pred[1])
            v.append(pred[2])

        x, y, v = np.array(x), np.array(y), np.array(v)

        kp_list = np.array(result['preds'][0]).reshape(-1)


        corr_slope_list, corr_slope_index_list, corr_tan_theta, corr_theta, corr_PT, corr_MT, corr_TL = cobb_correct(kp_list, res_idx, anomaly_matrix)

        corr_PT_str = str(corr_PT) + '°'
        corr_MT_str = str(corr_MT) + '°'
        corr_TL_str = str(corr_TL) + '°'
        cobb_corrected_dict[img_name + '-PT'] = corr_PT_str
        cobb_corrected_dict[img_name + '-MT'] = corr_MT_str
        cobb_corrected_dict[img_name + '-TL'] = corr_TL_str

        cobb_corrected.append([corr_PT, corr_MT, corr_TL])


        print("Image Name:", img_name)
        print("Corrected Slope List:", corr_slope_list)
        print("Corrected Slope Index List:", corr_slope_index_list)

        for sk in skeleton:

            if np.all(v[sk] > 0):

                if anomaly_matrix[res_idx][sk[0]] == True:
                    pass
                else:
                    # 画点之间的连接线
                    plt.plot(x[sk], y[sk], linewidth=1, color='#9EFF00')


        # 画点
        for i, (xi, yi, zi) in enumerate(zip(x, y, v)):
            if zi > 0:
                kpt_id = dataset_info['keypoint_info'][i]['id']
                color = [c / 255.0 for c in dataset_info['keypoint_info'][i]['color']]
                # color = '#' + self.rgb2hex((color[0], color[1], color[2])).replace('0x','').upper()
                plt.text(xi, yi, str(kpt_id), fontsize=3, color=[0, 0, 0])
                plt.plot(xi, yi, 'o', markersize=2, markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.3)

                line_length = 160 #法向量长度

                assert len(corr_slope_index_list)==4

                #画斜率线
                for k in corr_slope_index_list:
                    if i == k:

                        slope = corr_slope_list[corr_slope_index_list.index(k)]
                        delta_x = line_length / 2 / np.sqrt(1 + slope ** 2)  # 根据斜率和长度计算增量
                        x1 = xi - delta_x
                        y1 = yi - slope * delta_x
                        x2 = xi + delta_x
                        y2 = yi + slope * delta_x

                        print("Line:", [x1, x2], [y1, y2])

                        plt.plot([x1, x2], [y1, y2], linewidth=0.5, color='red')




        plt.text(300, 300, corr_MT, fontdict={'family': 'serif', 'size': 16, 'color': '#C00000'}, ha='center',
                 va='center')

        plt.savefig(os.path.join(cobb_visual_path, img_name.replace('.jpg', '_' + str(res_idx) + '.jpg')), dpi=600,
                            bbox_inches='tight')
        plt.close()

        print("="*20)

    cobb_json_path = os.path.join(cobb_visual_path, 'cobb')
    if not os.path.exists(cobb_json_path):
        os.makedirs(cobb_json_path)
    #将字典转换为JSON格式的字符串
    cobb_uncorrected_dict = json.dumps(cobb_uncorrected_dict)
    cobb_corrected_dict = json.dumps(cobb_corrected_dict)


    with open(os.path.join(cobb_json_path, 'cobb_uncorrected.json'), 'w', encoding='utf-8') as json_file:
        json_file.write(cobb_uncorrected_dict)
    with open(os.path.join(cobb_json_path, 'cobb_corrected.json'), 'w', encoding='utf-8') as json_file:
        json_file.write(cobb_corrected_dict)



    all_angle_mae = calculate_cobb_mae(cobb_true, cobb_corrected)

    PT_mae, MT_mae, TL_mae = all_angle_mae[0], all_angle_mae[1], all_angle_mae[2]

    smape = symmetric_mean_absolute_percentage(cobb_true, cobb_corrected)

    cmae = circular_mean_absolute_error(cobb_true, cobb_corrected)

    return cobb_uncorrected, cobb_corrected, PT_mae, MT_mae, TL_mae, smape, cmae


