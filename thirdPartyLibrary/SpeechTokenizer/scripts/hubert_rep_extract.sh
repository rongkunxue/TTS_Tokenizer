CONFIG="config/spt_base_cfg.json"
REP_DIR="/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/rep"
EXTS="wav"
SPLIT_SEED=0
VALID_SET_SIZE=1500



CUDA_VISIBLE_DEVICES=1 python scripts/hubert_rep_extract.py\
    --config ${CONFIG}\
    --rep_dir ${REP_DIR}\
    --exts ${EXTS}\
    --split_seed ${SPLIT_SEED}\
    --valid_set_size ${VALID_SET_SIZE}