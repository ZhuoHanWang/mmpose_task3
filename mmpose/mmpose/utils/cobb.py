import numpy as np
import json
import math


def exist_single_curve(i, j, tan):
    is_increasing = all(tan[k] <= tan[k + 1] for k in range(i, j))  # 检查从 i 到 j 是否递增
    if is_increasing:
        return True

    is_decreasing = all(tan[k] >= tan[k + 1] for k in range(i, j))  # 检查从 i 到 j 是否递减
    if is_decreasing:
        return True

    return False

def get_angle(keypoints):
    # 计算每个点连接后的vector
    kp_vec = []
    for i in range(16):
        x1 = keypoints[i][0]
        y1 = keypoints[i][1]
        x2 = keypoints[i + 1][0]
        y2 = keypoints[i + 1][1]
        kp_vec.append(np.array([x2 - x1, y2 - y1]))
    kp_vec.append(kp_vec[15])  # 最后一个点和倒数第二个点视为同一个


    # 计算垂直向量
    kp_vec = [np.array([-vec[1], vec[0]]) for vec in kp_vec]

    # 计算每个点的 tanθ和角度
    tan_theta = [vec[1] / vec[0] for vec in kp_vec]


    radian_value = map(math.atan, tan_theta)  # 正切转弧度值
    theta = list(map(math.degrees, radian_value))  # 弧度值转角度值

    # Angle matrix
    angle = np.zeros([17, 17])
    for i in range(17):
        for j in range(17):
            angle[i, j] = abs(theta[i] - theta[j])


    return tan_theta, theta, angle


def cobb_caculate(kp_list):

    keypoints = []
    for i in range(17):
        x = kp_list[i * 3]
        y = kp_list[i * 3 + 1]
        keypoint = np.array([x, y])
        keypoints.append(keypoint)

    tan_theta, theta, angle = get_angle(keypoints)


    delta = np.zeros((17, 17))
    for i in range(17):
        for j in range(17):
            if exist_single_curve(i, j, tan_theta) and i < j:
                delta[i, j] = 1

    new_angle = angle * delta


    cobb_tuple = ()
    # 计算三个cobb角
    # a1
    a1 = np.max(new_angle)
    a1_index = np.unravel_index(np.argmax(new_angle), new_angle.shape)
    cobb_tuple += ("a1", a1, a1_index)
    # print('a1: ', a1, a1_index)

    # a2
    top1 = a1_index[0]
    bottom1 = a1_index[1]
    a2_delta = np.zeros((17, 17))
    l1 = range(top1 + 1)
    l2 = range(bottom1, 17)
    for i in l1:
        a2_delta[i, top1] = 1
    for j in l2:
        a2_delta[bottom1, j] = 1
    a2_angle = a2_delta * angle * delta

    a2 = np.max(a2_angle)
    a2_index = np.unravel_index(np.argmax(a2_angle), a2_angle.shape)
    cobb_tuple += ("a2", a2, a2_index)
    # print('a2: ', a2, a2_index)

    # a3
    top2 = a2_index[0]
    bottom2 = a2_index[1]
    a3_delta = np.zeros((17, 17))
    if top2 < top1:  # above
        l3 = range(top2 + 1)
        for i in l3:
            a3_delta[i, top2] = 1
        for j in l2:
            a3_delta[bottom1, j] = 1
        a3_angle = a3_delta * angle * delta
        a3 = np.max(a3_angle)
        a3_index = np.unravel_index(np.argmax(a3_angle), a3_angle.shape)
        # print('a3: ',a3, a3_index)
    else:
        l4 = range(bottom2, 17)
        for i in l1:
            a3_delta[i, top1] = 1
        for j in l4:
            a3_delta[bottom2, j] = 1
        a3_angle = a3_delta * angle * delta
        a3 = np.max(a3_angle)
        a3_index = np.unravel_index(np.argmax(a3_angle), a3_angle.shape)
    cobb_tuple += ("a3", a3, a3_index)

    final_cobb = max([cobb_tuple[1], cobb_tuple[4], cobb_tuple[7]])

    return round(final_cobb,2)


if __name__ == '__main__':
    # 加载数据
    with open('D:/Datasets/spine/annotations/sample/spine_keypoints_v2_val.json', 'r') as file:
        data = json.load(file)

    data = data['annotations'][0]

    # keypoints = [round(num,0) for num in data['keypoints']]
    kp_list = data['keypoints']

    out = cobb_caculate(kp_list)
    print(out)