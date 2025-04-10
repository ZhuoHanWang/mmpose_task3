from mmpose.apis import MMPoseInferencer
import os
import json
import argparse
from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with MMPose')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:2',
        help='Device to use')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 获取配置文件的最后一级文件名（不含后缀）
    config_basename = os.path.splitext(os.path.basename(args.config))[0]
    result_filename = f"{config_basename}_result.json"
    
    # 获取数据集配置
    data_root = cfg.data_root
    ann_file = os.path.join(data_root, cfg.val_dataloader.dataset.ann_file)
    
    # 固定输出目录，不再创建子目录
    output_dir = '/pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result'
    os.makedirs(output_dir, exist_ok=True)
    
    # 直接在输出目录下保存结果文件
    result_file = os.path.join(output_dir, result_filename)
    
    # 创建推理器
    inferencer = MMPoseInferencer(
        pose2d=args.config,
        pose2d_weights=args.checkpoint,
        device=args.device
    )
    
    # 加载标注文件
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # 获取图片路径，使用配置文件中的data_prefix
    img_prefix = os.path.join(data_root, cfg.val_dataloader.dataset.data_prefix.get('img', 'images'))
    image_files = [os.path.join(img_prefix, img_info['file_name']) 
                  for img_info in annotations['images']]
    
    # 结果列表
    results = []
    
    print(f'开始推理，共 {len(image_files)} 张图像...')
    
    # 遍历图片进行推理
    for idx, img_path in enumerate(image_files):
        print(f'处理图像 {idx+1}/{len(image_files)}')
        
        # 执行推理
        result_generator = inferencer(img_path)
        result = next(result_generator)
        
        # 获取关键点和分数
        pred_instance = result['predictions'][0][0]  # 获取第一个预测实例
        keypoints = pred_instance['keypoints']       # 关键点坐标
        scores = pred_instance['keypoint_scores']    # 关键点分数
        
        # 组合关键点和分数
        combined_keypoints = []
        for kpt, score in zip(keypoints, scores):
            combined_keypoints.append([
                float(kpt[0]),    # x坐标
                float(kpt[1]),    # y坐标
                float(score)      # 置信度分数
            ])
        
        # 构建结果字典
        result_item = {
            'image_paths': [img_path],  # 改为列表格式
            'preds': [combined_keypoints]  # 嵌套列表格式
        }
        results.append(result_item)
    
    # 保存结果
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f'推理完成，结果已保存至 {result_file}')

if __name__ == '__main__':
    main() 