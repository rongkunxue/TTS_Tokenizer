CONFIG="config/spt_base_cfg.json"
AUDIO_DIR="/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-100"
REP_DIR="/root/Github/TTS_Tokenizer/thirdPartyLibrary/SpeechTokenizer/rep"
EXTS="wav"
SPLIT_SEED=0
VALID_SET_SIZE=1500



CUDA_VISIBLE_DEVICES=0 python scripts/hubert_rep_extract.py\
    --config ${CONFIG}\
    --audio_dir ${AUDIO_DIR}\
    --rep_dir ${REP_DIR}\
    --exts ${EXTS}\
    --split_seed ${SPLIT_SEED}\
    --valid_set_size ${VALID_SET_SIZE}