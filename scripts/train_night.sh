
dataset1=$1
workspace1=$2
gpu_id=0
#audio_extractor='ave' # deepspeech, esperanto, hubert
audio_extractor='deepspeech' # deepspeech, esperanto, hubert


pretrain_project_path="out/Kplane"

pretrain_face_path=${pretrain_project_path}/chkpnt_ema_face_latest.pth
pretrain_mouth_path=${pretrain_project_path}/chkpnt_ema_mouth_latest.pth

# n_views=500 # 20s
n_views=275 # 10s
#n_views=150 # 5s


export CUDA_VISIBLE_DEVICES=$gpu_id
foldpath=output/Kdap
name=Kdap

dataset1=data/pretrain/1
workspace1=${foldpath}/1${name}
#CUDA_LAUNCH_BLOCKING=1
python new_train_face_KTF.py --type face -s $dataset1 -m $workspace1 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 14000 --sh_degree 1 --N_views $n_views
python synthesize_fuse.py -s $dataset1 -m $workspace1 --eval --audio_extractor $audio_extractor --dilate
python metrics.py $workspace1/test/ours_None/renders/out.mp4 $workspace1/test/ours_None/gt/out.mp4
python synthesize_fuse.py -S $dataset1 -m $workspace1 --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy

dataset2=data/pretrain/2
workspace2=${foldpath}/2${name}
#CUDA_LAUNCH_BLOCKING=1
#python new_train_face_K.py --type face -s $dataset2 -m $workspace2 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 14000 --sh_degree 1 --N_views $n_views
#python synthesize_fuse.py -s $dataset2 -m $workspace2 --eval --audio_extractor $audio_extractor --dilate
#python metrics.py $workspace2/test/ours_None/renders/out.mp4 $workspace2/test/ours_None/gt/out.mp4
#python synthesize_fuse.py -S $dataset2 -m $workspace2 --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy

dataset3=data/pretrain/0
workspace3=${foldpath}/0${name}
#python new_train_face_K --type face -s $dataset3 -m $workspace3 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 14000 --sh_degree 1 --N_views $n_views
#python synthesize_fuse.py -s $dataset3 -m $workspace3 --eval --audio_extractor $audio_extractor --dilate
#python metrics.py $workspace3/test/ours_None/renders/out.mp4 $workspace3/test/ours_None/gt/out.mp4
#python synthesize_fuse.py -S $dataset3 -m $workspace3 --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy

dataset4=data/Lieu
workspace4=${foldpath}/Lieu${name}
#python new_train_face_K.py --type face -s $dataset4 -m $workspace4 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 14000 --sh_degree 1 --N_views $n_views
#python synthesize_fuse.py -s $dataset4 -m $workspace4 --eval --audio_extractor $audio_extractor --dilate
#python metrics.py $workspace4/test/ours_None/renders/out.mp4 $workspace4/test/ours_None/gt/out.mp4
#python synthesize_fuse.py -S $dataset4 -m $workspace4 --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy

dataset5=data/cnn2
workspace5=${foldpath}/cnn2${name}
python new_train_face_K.py --type face -s $dataset5 -m $workspace5 --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 14000 --sh_degree 1 --N_views $n_views
python synthesize_fuse.py -s $dataset5 -m $workspace5 --eval --audio_extractor $audio_extractor --dilate
python metrics.py $workspace5/test/ours_None/renders/out.mp4 $workspace5/test/ours_None/gt/out.mp4
python synthesize_fuse.py -S $dataset5 -m $workspace5 --dilate --use_train --audio /root/autodl-tmp/InsTaG/data/5/aud10s.npy

#python new_pretrain_face.py -s data -m output/TFtri_pre --type face --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor deepspeech --iterations 50000

