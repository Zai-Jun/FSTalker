dataset=$1
workspace=$2
gpu_id=0
audio_extractor='deepspeech' # deepspeech, esperanto, hubert

export CUDA_VISIBLE_DEVICES=$gpu_id

python new_pretrain_face_K.py -s $dataset -m $workspace --type face --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --iterations 50000
#python pretrain_face.py -s data/pretrain -m output/tri_pretrain --type face --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor 'deepspeech' --iterations 30000


# python pretrain_mouth.py -s $dataset -m $workspace --type mouth --init_num 5000 --audio_extractor $audio_extractor  --iterations 30000
                        #python pretrain_mouth.py -s data/pretrain -m output/test0 --type mouth --init_num 5000 --audio_extractor deepspeech --iterations 30000