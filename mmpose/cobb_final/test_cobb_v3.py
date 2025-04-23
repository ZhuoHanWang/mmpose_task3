import argparse
import os
import os.path as osp
import json
import mmcv
import sys
from datetime import datetime
#from mmcv import Config, DictAction

# 导入Cobb角度计算和可视化模块
from cobb_final.cobb_v3 import show_cobb

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='mmpose test model')

    # 预测结果文件路径，包含关键点预测结果
    parser.add_argument('--result', help='result json file path')
    args = parser.parse_args()
    return args

def main():
    """主函数"""
    args = parse_args()
    
    # 获取result文件名（不含后缀）
    result_name = os.path.splitext(os.path.basename(args.result))[0]
    
    # 读取预测结果JSON文件
    with open(args.result, 'r') as f:
        outputs = json.load(f)
    
    # 修改保存路径，使用result_name作为子文件夹
    base_save_path = '/pangyan/wzh/mmpose_task/task3/model_result_cobb/cobb_final_visual'
    save_path = os.path.join(base_save_path, result_name)
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(save_path, f"{result_name}.log")
    
    # 将终端输出重定向到日志文件
    sys.stdout = open(log_file, 'w')
    
    print(f"开始处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"处理文件: {args.result}")
    print(f"日志保存路径: {log_file}")
    print("-" * 50)
    
    # 计算Cobb角度并生成可视化结果
    (cobb_uncorrected, cobb_corrected, 
     corrected_PT_mae, corrected_MT_mae, corrected_TL_mae, corrected_smape, corrected_cmae,
     uncorrected_PT_mae, uncorrected_MT_mae, uncorrected_TL_mae, uncorrected_smape, uncorrected_cmae) = show_cobb(
        results=outputs, 
        annFile='/pangyan/wzh/mmpose_task/task3/data/annotations/sample/spine_keypoints_rgb_1_v2_val.json',
        detector_path='/root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/autoencoder',
        save_path=save_path
    )
    
    # 恢复标准输出
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    # 在终端显示简要信息
    print(f"\n处理文件: {result_name}")
    print(f"日志已保存到: {log_file}")
    print("\n评估结果:")
    print("-" * 50)
    print("校正后:")
    # 计算校正后的平均值
    corrected_avg_mae = (corrected_MT_mae + corrected_PT_mae + corrected_TL_mae) / 3
    print(f"AVG MAE: {corrected_avg_mae:.2f}°")
    print(f"MT MAE: {corrected_MT_mae:.2f}°")
    print(f"PT MAE: {corrected_PT_mae:.2f}°")
    print(f"TL MAE: {corrected_TL_mae:.2f}°")
    print(f"SMAPE: {corrected_smape:.2f}%")
    print(f"CMAE: {corrected_cmae:.2f}°")
    print("-" * 50)
    print("校正前:")
    # 计算校正前的平均值
    uncorrected_avg_mae = (uncorrected_MT_mae + uncorrected_PT_mae + uncorrected_TL_mae) / 3
    print(f"AVG MAE: {uncorrected_avg_mae:.2f}°")
    print(f"MT MAE: {uncorrected_MT_mae:.2f}°")
    print(f"PT MAE: {uncorrected_PT_mae:.2f}°")
    print(f"TL MAE: {uncorrected_TL_mae:.2f}°")
    print(f"SMAPE: {uncorrected_smape:.2f}%")
    print(f"CMAE: {uncorrected_cmae:.2f}°")

if __name__ == '__main__':
    main()