import os
import cv2
import json
import numpy as np
import glob
import torch
import subprocess
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer
from mmpose.utils import register_all_modules
from mmpose.utils.cobb import cobb_caculate

def calculate_custom_cobb(keypoints, keypoint_scores):
    """独立的Cobb角计算函数，读取了模型推理的关键点结果，用于验证和对比local_visualizer计算的Cobb角结果"""
    # 准备关键点数据
    kp_list = []
    valid_mask = keypoint_scores > 0.3  # 使用相同的置信度阈值

    # 将关键点转换为cobb_calculate需要的格式
    for kp, score in zip(keypoints, keypoint_scores):
        kp_list.extend([kp[0], kp[1], float(score)])
    
    # 计算Cobb角
    cobb_angle = cobb_caculate(np.array(kp_list))
    return cobb_angle

def process_single_image(img_path, estimator, visualizer, save_dir, results_dict):
    """处理单张图片的函数"""
    print(f"\nProcessing image: {os.path.basename(img_path)}")
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f'Warning: Image {img_path} not found or corrupted, skipping...')
        return
    
    # 推理
    predictions = inference_topdown(estimator, img)
    pred_instance = predictions[0]
    
    # 获取关键点信息
    keypoints = pred_instance.pred_instances.keypoints[0]  # 形状为(N, 2)的数组
    keypoint_scores = pred_instance.pred_instances.keypoint_scores[0]  # 形状为(N,)的数组
    
    # 计算自定义的Cobb角并保存结果
    custom_cobb = calculate_custom_cobb(keypoints, keypoint_scores)
    print(f"Custom calculated Cobb angle: {custom_cobb:.1f}°")
    
    # 将结果保存到字典中
    results_dict[os.path.basename(img_path)] = {
        'cobb_angle': float(custom_cobb),
        'keypoints': keypoints.tolist(),
        'keypoint_scores': keypoint_scores.tolist()
    }
    
    # 设置图像路径
    pred_instance.set_metainfo({'img_path': img_path})
    
    # 绘制关键点和骨架
    visualization = visualizer.add_datasample(
        'result',
        img,
        data_sample=pred_instance,
        show=False,
        draw_bbox=False,  # 不显示边界框
        kpt_thr=-1.0,     # 使用负值确保所有关键点都被绘制
        show_kpt_idx=True,
        skeleton_style='mmpose',
    )

    # 保存可视化结果
    save_path = os.path.join(save_dir, os.path.basename(img_path).replace('.png', '_vis.jpg'))
    if visualization is not None:
        cv2.imwrite(save_path, visualization)
    else:
        print(f"Warning: Visualization is None for {os.path.basename(img_path)}")

def get_available_gpus():
    """获取单个最空闲的GPU"""
    try:
        # 使用nvidia-smi命令获取GPU使用情况
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        memory_used = subprocess.check_output(cmd.split()).decode('utf-8').strip().split('\n')
        
        # 找到显存占用最小的GPU
        memory_used = [float(x) for x in memory_used]
        min_mem_gpu = memory_used.index(min(memory_used))
        return [min_mem_gpu]
    except:
        return [0]  # 如果出现错误，默认使用GPU 0

def main():
    # 模型配置
    model_name = 'vipnas_r50'  # 模型简称
    config = '/root/task3/mmpose/configs/spine_2d_keypoint/topdown_heatmap/spine/td-hm_vipnas-res50_8x64-200e_spine-256x256.py'
    checkpoint = '/root/task3/mmpose/work_dirs/td-hm_vipnas-res50_8x64-200e_spine-256x256/best_PCK@0.01_epoch_144.pth'
    
    # 注册所有模块
    register_all_modules()
    
    # 获取可用的GPU
    available_gpus = get_available_gpus()
    print(f"使用GPU: {available_gpus}")
    
    # 初始化主GPU上的模型
    print("Initializing model...")
    main_device = f'cuda:{available_gpus[0]}'
    estimator = init_model(
        config, 
        checkpoint,
        device=main_device
    )
    
    # 准备可视化器
    visualizer = PoseLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        cobb_path='/root/task3/mmpose/',  # 基础路径
        config_name=model_name,  # 使用模型简称作为子目录
        draw_bbox=False,
        radius=15,
        alpha=1.0,
        line_width=8
    )
    visualizer.set_dataset_meta(estimator.dataset_meta)

    # 创建结果保存目录
    save_dir = os.path.join('/root/task3/mmpose/vis_results', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有X光图片
    image_pattern = '/root/task3/data/images/*xray*.png'
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print("No X-ray images found!")
        return
    
    print(f"\nFound {len(image_files)} X-ray images to process")
    
    # 创建结果字典
    results_dict = {}
    
    # 处理每张图片
    for i, img_path in enumerate(image_files, 1):
        print(f"\nProcessing image {i}/{len(image_files)}: {os.path.basename(img_path)}")
        process_single_image(img_path, estimator, visualizer, save_dir, results_dict)
    
    # 保存汇总结果
    cobb_save_dir = f'/root/task3/mmpose/cobb/{model_name}'
    os.makedirs(cobb_save_dir, exist_ok=True)
    summary_path = os.path.join(cobb_save_dir, 'all_results.json')
    with open(summary_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    # 提取并保存cobb角度
    cobb_dict = {}
    for img_name, result in results_dict.items():
        cobb_dict[img_name] = result['cobb_angle']
        
    # 将所有cobb角度保存到单个JSON文件
    cobb_only_path = os.path.join(cobb_save_dir, 'cobb_angles.json')
    with open(cobb_only_path, 'w') as f:
        json.dump(cobb_dict, f, indent=4)
    print("\nProcessing complete!")
    print(f"Results saved in:")
    print(f"- Visualizations: {save_dir}")
    print(f"- Cobb angles: /root/task3/mmpose/cobb/{model_name}/")

if __name__ == '__main__':
    main()
