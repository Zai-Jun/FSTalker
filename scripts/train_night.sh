dataset=$1
workspace=$2
gpu_id=$3
audio_extractor='deepspeech' # deepspeech, esperanto, hubert
gpu_id=0
export CUDA_VISIBLE_DEVICES=$gpu_id

# 临时清空当前终端的动态库查找路径
export LD_LIBRARY_PATH=""
# 重新验证链接状态（预期两个库都会指向 anaconda3 内部目录）
ldd $(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_cuda.so'))") | grep cublas

#python train_face_tf_moe.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor

python train_face_FLAME.py -s VHAPdata/1 -m output/UVtest1 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor
python synthesize_fuse.py -s VHAPdata/1 -m output/UVtest1 --eval --audio_extractor $audio_extractor

#python train_face_tf_moe.py -s data/pretrain/2 -m output/moeAF_test2 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor
#python synthesize_fuse.py -s data/pretrain/2 -m output/moeAF_test2 --eval --audio_extractor $audio_extractor

#python train_face_tf_moe.py -s data/pretrain/0 -m output/moeAF_test0 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor
#python synthesize_fuse.py -s data/pretrain/0 -m output/moeAF_test0 --eval --audio_extractor $audio_extractor

#python metrics.py output/UVtest1/test/ours_None/renders/out.mp4 output/UVtest1/test/ours_None/gt/out.mp4
#python metrics.py output/Ffix_3k_1/test/ours_None/renders/out.mp4 output/Ffix_3k_1/test/ours_None/gt/out.mp4
#python metrics.py output/moeAF_test2/test/ours_None/renders/out.mp4 output/moeAF_test2/test/ours_None/gt/out.mp4
