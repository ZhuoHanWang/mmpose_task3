import argparse
import os
import os.path as osp
import json
import mmcv
import sys
from datetime import datetime
#from mmcv import Config, DictAction

# 导入Cobb角度计算和可视化模块
from cobb_final.cobb_v2 import show_cobb

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
    
    save_path = '/pangyan/wzh/mmpose_task/task3/model_result_cobb/cobb_final_visual'
    
    # 设置日志文件路径
    log_file = os.path.join(save_path, f"{result_name}.log")
    
    # 将终端输出重定向到日志文件
    sys.stdout = open(log_file, 'w')
    
    print(f"开始处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"处理文件: {args.result}")
    print(f"日志保存路径: {log_file}")
    print("-" * 50)
    
    # 计算Cobb角度并生成可视化结果
    cobb_uncorrected, cobb_corrected, PT_mae, MT_mae, TL_mae, smape, cmae = show_cobb(
        results=outputs, 
        annFile='/pangyan/wzh/mmpose_task/task3/data/annotations/sample/spine_keypoints_rgb_1_v2_val.json',
        detector_path='/root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/autoencoder',
        save_path=save_path
    )
    
    # 恢复标准输出
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    # 在终端只显示简要信息
    print(f"\n处理文件: {result_name}")
    print(f"日志已保存到: {log_file}")
    print("\n评估结果:")
    print("-" * 30)
    print(f"PT平均绝对误差: {PT_mae:.2f}°")
    print(f"MT平均绝对误差: {MT_mae:.2f}°")
    print(f"TL平均绝对误差: {TL_mae:.2f}°")
    print(f"SMAPE: {smape:.2f}%")
    print(f"CMAE: {cmae:.2f}°")

if __name__ == '__main__':
    main()