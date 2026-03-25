dataset=$1
workspace=$2
gpu_id=0
#audio_extractor='ave' # deepspeech, esperanto, hubert
audio_extractor='deepspeech' # deepspeech, esperanto, hubert


pretrain_project_path="out/trimlp"

pretrain_face_path=${pretrain_project_path}/chkpnt_ema_face_latest.pth
pretrain_mouth_path=${pretrain_project_path}/chkpnt_ema_mouth_latest.pth

# n_views=500 # 20s
n_views=250 # 10s
# n_views=150 # 5s


export CUDA_VISIBLE_DEVICES=$gpu_id

dataset=data/pretrain/0
workspace=output/etriTF/0etriATF
python synthesize_fuse.py -S $dataset -m $workspace --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy


dataset=data/pretrain/1
workspace=output/etriTF/1etriATF
python synthesize_fuse.py -S $dataset -m $workspace --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy


dataset=data/pretrain/2
workspace=output/etriTF/2etriATF
python synthesize_fuse.py -S $dataset -m $workspace --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy

dataset=data/Lieu
workspace=output/etriTF/LieuetriATF
python synthesize_fuse.py -S $dataset -m $workspace --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy