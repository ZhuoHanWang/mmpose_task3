from mmpose.apis import MMPoseInferencer

img_path = 'D:/Datasets/spine-unimodal/images/00001_rgb_1.jpg'   # 将img_path替换给你自己的路径

# 使用模型别名创建推理器
inferencer = MMPoseInferencer(pose2d='./configs/spine_2d_keypoint/topdown_heatmap/spine/td-hm_hrnet-w48_8xb32-210e_spine-256x256.py',
                              pose2d_weights='./work_dirs/td-hm_hrnet-w48_8xb32-210e_spine-256x256/best_PCK_epoch_36.pth',
                              )

# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
result_generator = inferencer(img_path, show=True)
result = next(result_generator)