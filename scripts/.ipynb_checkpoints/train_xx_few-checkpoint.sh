dataset=$1
workspace=$2
gpu_id=$3
#audio_extractor='ave' # deepspeech, esperanto, hubert
audio_extractor='deepspeech' # deepspeech, esperanto, hubert


pretrain_project_path="output/old/pe12/"

pretrain_face_path=${pretrain_project_path}/chkpnt_ema_face_latest.pth
pretrain_mouth_path=${pretrain_project_path}/chkpnt_ema_mouth_latest.pth

# n_views=500 # 20s
#n_views=250 # 10s
 n_views=125 # 5s


export CUDA_VISIBLE_DEVICES=$gpu_id

python train_face.py --type face -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 10000 --sh_degree 1 --N_views $n_views
python train_mouth.py --type mouth -s $dataset -m $workspace --audio_extractor $audio_extractor --pretrain_path $pretrain_mouth_path --init_num 5000 --iterations 10000 --sh_degree 1 --N_views $n_views
python train_fuse_con.py -s $dataset -m $workspace --opacity_lr 0.001 --audio_extractor $audio_extractor --iterations 2000 --sh_degree 1 --N_views $n_views

python synthesize_fuse.py -s $dataset -m $workspace --eval --audio_extractor $audio_extractor --dilate
python metrics.py $workspace/test/ours_None/renders/out.mp4 $workspace/test/ours_None/gt/out.mp4


#python train_face.py --type face -s data/5 -m output/8mlp51 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor 'deepspeech' --pretrain_path output/8mlp0/chkpnt_ema_face_latest.pth --iterations 10000 --sh_degree 1 --N_views 125
#python train_mouth.py --type mouth -s data/5 -m output/8mlp51 --audio_extractor deepspeech --pretrain_path output/8mlp0/chkpnt_ema_mouth_latest.pth --init_num 5000 --iterations 10000 --sh_degree 1 --N_views 125