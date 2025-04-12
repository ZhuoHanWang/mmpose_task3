#!/bin/bash

python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/simcc_res50_8xb64-200e_coco-256×256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/td-hm_hourglass52_8xb32-200e_spine-256x256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/td-hm_hrformer-base_8xb32-200e_spine-256x256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/td-hm_hrnet-w48_8xb32-200e_spine-256x256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/td-hm_hrnetv2-w48_1xb16-200e_spine-256×256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/td-hm_litehrnet-30_200e_spine-256x256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/td-hm_vipnas-mbv3_8xb64-200e_spine-256x256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task3/model_result_cobb/model_result/td-hm_vipnas-res50_8x64-200e_spine-256x256_result.json && \
python /root/WZH_ONLY_CODE_HERE/task3/mmpose/cobb_final/test_cobb_v3.py --result /pangyan/wzh/mmpose_task/task4/model_result_cobb/model_result/sharpose_result.json