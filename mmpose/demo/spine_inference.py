import os
import cv2
import json
import numpy as np
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

def main():
    # 配置和路径
    config = '/root/task3/mmpose/configs/spine_2d_keypoint/topdown_heatmap/spine/simcc_res50_8xb64-200e_coco-256×256.py'
    checkpoint = '/root/task3/mmpose/work_dirs/simcc_res50_8xb64-200e_coco-256×256/best_PCK@0.01_epoch_104.pth'
    img_path = '/root/task3/data/images/00001_xray_1.png'
    
    # 注册所有模块
    register_all_modules()
    
    # 初始化模型
    estimator = init_model(
        config, 
        checkpoint,
        device='cuda:0'
    )
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Image {img_path} not found')

    # 推理
    predictions = inference_topdown(estimator, img)
    pred_instance = predictions[0]
    
    # 获取关键点信息
    keypoints = pred_instance.pred_instances.keypoints[0]  # 形状为(N, 2)的数组
    keypoint_scores = pred_instance.pred_instances.keypoint_scores[0]  # 形状为(N,)的数组
    
    # 计算自定义的Cobb角
    custom_cobb = calculate_custom_cobb(keypoints, keypoint_scores)
    print("\n=== Custom Cobb Angle Calculation ===")
    print(f"Custom calculated Cobb angle: {custom_cobb:.1f}°")
    
    # 打印关键点信息
    print("\n=== Keypoint Information ===")
    for i, (kp, score) in enumerate(zip(keypoints, keypoint_scores)):
        print(f"Keypoint {i}: position=({kp[0]:.1f}, {kp[1]:.1f}), score={score:.3f}")
        
    # # 设置图像元信息
    # pred_instance.set_metainfo({
    #     'img_path': img_path,
    #     'flip': False,
    #     'bbox_score': 1.0,
    #     'bbox_center': [img.shape[1]//2, img.shape[0]//2],
    #     'bbox_scale': [img.shape[1], img.shape[0]],
    # })

    # 准备可视化
    visualizer = PoseLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        cobb_path='/root/task3/mmpose/',  # 设置Cobb角度结果保存路径
        draw_bbox=False,
        radius=15,  # 进一步增大关键点半径
        alpha=1.0,  # 增加不透明度
        line_width=8  # 进一步增大线宽
    )
    visualizer.set_dataset_meta(estimator.dataset_meta)

    # 创建结果保存目录
    save_dir = os.path.join('/root/task3/mmpose/vis_results')
    os.makedirs(save_dir, exist_ok=True)

    # 绘制关键点和骨架
    # 设置图像路径
    pred_instance.set_metainfo({'img_path': img_path})
    
    # 打印所有关键点信息
    print("\n=== All Keypoint Information ===")
    for i, (kp, score) in enumerate(zip(keypoints, keypoint_scores)):
        print(f"Keypoint {i}: position=({kp[0]:.1f}, {kp[1]:.1f}), score={score:.3f}")

    # 确保所有关键点都被绘制
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

    # 保存结果和日志
    save_path = os.path.join(save_dir, os.path.basename(img_path).replace('.png', '_vis.jpg'))
    cv2.imwrite(save_path, visualization)
    print(f'\nVisualization saved to {save_path}')
    
    # 读取local_visualizer计算的Cobb角结果
    cobb_result_path = os.path.join('/root/task3/mmpose/cobb/default_config', 
                                   os.path.basename(img_path).replace('.png', '.json'))
    try:
        with open(cobb_result_path, 'r') as f:
            visualizer_cobb = json.loads(f.read())
            print("\n=== Local Visualizer Cobb Angle ===")
            print(f"Visualizer calculated Cobb angle: {visualizer_cobb[os.path.basename(img_path)]}")
    except FileNotFoundError:
        print(f"\nWarning: Visualizer Cobb angle result not found at {cobb_result_path}")

if __name__ == '__main__':
    main()
