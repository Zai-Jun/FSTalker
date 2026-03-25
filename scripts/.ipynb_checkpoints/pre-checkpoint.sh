python data_utils/process.py data/em/em.mp4
python data_utils/split.py data/em/em.mp4 

export PYTHONPATH=./data_utils/easyportrait 
python ./data_utils/easyportrait/create_teeth_mask.py ./data/em

conda activate sapiens_lite
bash ./data_utils/sapiens/run.sh ./data/em