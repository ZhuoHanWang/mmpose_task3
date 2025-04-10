import argparse
import os
import os.path as osp
import json
import mmcv
#from mmcv import Config, DictAction

# 导入Cobb角度计算和可视化模块
from cobb_final.cobb import show_cobb

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='mmpose test model')

    # 预测结果文件路径，包含关键点预测结果
    parser.add_argument('--result', help='result json file path')
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    # 获取命令行参数
    args = parse_args()
    
    # 读取预测结果JSON文件
    with open(args.result, 'r') as f:
        outputs = json.load(f)
    
    # 计算Cobb角度并生成可视化结果
    # results: 预测的关键点结果
    # annFile: 数据集标注文件，包含真实关键点信息
    # detector_path: 异常检测模型路径
    # save_path: 可视化结果保存路径
    cobb_uncorrected, cobb_corrected, PT_mae, MT_mae, TL_mae, smape, cmae = show_cobb(
        results=outputs, 
        annFile='/pangyan/wzh/mmpose_task/task3/data/annotations/sample/spine_keypoints_rgb_1_v2_val.json',
        detector_path='/root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/autoencoder',
        save_path='/pangyan/wzh/mmpose_task/task3/model_result_cobb/cobb_final_visual'
    )

    # 打印评估指标
    print("PT MAE:", PT_mae)    # 近端胸椎角度平均绝对误差
    print("MT MAE:", MT_mae)    # 主胸椎角度平均绝对误差
    print("TL MAE:", TL_mae)    # 胸腰椎角度平均绝对误差
    print("SMAPE:", smape)      # 对称平均绝对百分比误差
    print("CMAE:", cmae)        # 环形平均绝对误差

if __name__ == '__main__':
    main()
