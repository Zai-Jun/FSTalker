#!/bin/bash

# 基础配置参数（统一维护，便于修改）
gpu_id=0
audio_extractor='deepspeech'  # deepspeech, esperanto, hubert
pretrain_project_path="output/woNC"
pretrain_face_path=${pretrain_project_path}/chkpnt_ema_face_latest.pth
n_views=250  # 10s (可选：500=20s, 150=5s)
foldpath=output/abl
name=woNC
audio_file="/root/autodl-tmp/InsTaG/data/Obama10s.npy"  # 统一音频文件路径

# 导出CUDA设备环境变量
export CUDA_VISIBLE_DEVICES=$gpu_id

# 定义数据集列表（格式："数据集目录 工作空间后缀"，按需添加/删除）
datasets=(
    "data/pretrain/0 0"
    "data/pretrain/1 1"
    "data/pretrain/2 2"
    "data/Lieu Lieu"
    "data/cnn2 cnn2"
)


# 循环遍历所有数据集，执行统一任务流程
for item in "${datasets[@]}"; do
    # 拆分数据集目录和工作空间后缀
    dataset=$(echo "$item" | awk '{print $1}')
    ws_suffix=$(echo "$item" | awk '{print $2}')

    # 构造工作空间路径
    workspace=${foldpath}/${ws_suffix}${name}

    # 打印当前执行进度（便于排查问题）
    echo -e "\n======================================"
    echo "Dataset：$dataset"
    echo "WorkSpace：$workspace"
    echo "======================================"

  python new_train_face_KTF.py --type face -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 14000 --sh_degree 1 --N_views $n_views

  python synthesize_fuse.py -s $dataset -m $workspace --eval --audio_extractor $audio_extractor --dilate
  python metrics.py $workspace/test/ours_None/renders/kp_oba.mp4 $workspace/test/ours_None/gt/kp_oba.mp4
  python synthesize_fuse.py -S $dataset -m $workspace --dilate --use_train --audio $audio_file


done

# 循环结束提示
echo -e "\nDone！"
