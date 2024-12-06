import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path
CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}
supports = {
    # CAM++ trained on 200k labeled speakers
    'iic/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    # CAM++ trained on a large-scale Chinese-English corpus
    'iic/speech_campplus_sv_zh_en_16k-common_advanced': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_en_common.pt',
    },
    # CAM++ trained on VoxCeleb
    'iic/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
}

model_id='iic/speech_campplus_sv_zh-cn_16k-common'
save_dir = os.path.join("pretrained", "pretrained/speech_campplus_sv_zh-cn_16k-common")
save_dir = pathlib.Path(save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
conf = supports["iic/speech_campplus_sv_zh-cn_16k-common"]
cache_dir = snapshot_download(
            model_id,
            revision=conf['revision'],
    )
cache_dir = pathlib.Path(cache_dir)
embedding_dir = save_dir / 'embeddings'
embedding_dir.mkdir(exist_ok=True, parents=True)
download_files = ['examples', conf['model_pt']]
for src in cache_dir.glob('*'):
    if re.search('|'.join(download_files), src.name):
        dst = save_dir / src.name
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        dst.symlink_to(src)
pretrained_model = save_dir / conf['model_pt']
pretrained_state = torch.load(pretrained_model, map_location='cpu')
model = conf['model']
embedding_model = dynamic_import(model['obj'])(**model['args'])
embedding_model.load_state_dict(pretrained_state)
device = torch.device('cuda')
embedding_model.to(device)
embedding_model.eval()


def compute_embedding(wav_file, save=True):
    obj_fs=16000
    wav, fs = torchaudio.load(wav_file)
    if fs != obj_fs:
        print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
        wav, fs = torchaudio.sox_effects.apply_effects_tensor(
            wav, fs, effects=[['rate', str(obj_fs)]]
        )
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    feat = feature_extractor(wav).unsqueeze(0).to(device)
    # compute embedding
    with torch.no_grad():
        embedding = embedding_model(feat).detach().squeeze(0).cpu().numpy()
    return embedding


examples_dir = save_dir / 'examples'
wav_path1, wav_path2 = list(examples_dir.glob('*.wav'))[0:2]

embedding1 = compute_embedding(wav_path1)
a=1

