from transformers import HubertModel,  Wav2Vec2FeatureExtractor
from pathlib import Path
import torchaudio
import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import os
from transformers import Wav2Vec2BertModel
from transformers import SeamlessM4TFeatureExtractor

def build_semantic_model(device):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)
    stat_mean_var = torch.load("/root/Github/Amphion/models/tts/maskgct/ckpt/wav2vec2bert_stats.pt")
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    return semantic_model, semantic_mean, semantic_std

@torch.no_grad()
def extract_features(processor,speech):
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"][0]
    attention_mask = inputs["attention_mask"][0]
    return input_features, attention_mask


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config/spt_base_cfg.json")
    parser.add_argument('--rep_dir', type=str, default="/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/rep")
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav,mp3')
    parser.add_argument('--split_seed', type=int, help="Random seed", default=0)
    parser.add_argument('--valid_set_size', type=float, default=1500)
    args = parser.parse_args()
    exts = args.exts.split(',')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = json.load(f)
    sample_rate = cfg.get('sample_rate')
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960",cache_dir="/checkpoint")
    # model = HubertModel.from_pretrained("facebook/hubert-base-ls960",cache_dir="checkpoint").eval().to(device)
    processor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    target_layer = cfg.get('semantic_model_layer')
    path1 = Path("/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-100")
    file_list = [
        str(file) for ext in exts 
        for path in [path1] 
        for file in path.glob(f'**/*.{ext}')
    ]
    if args.valid_set_size != 0 and args.valid_set_size < 1:
        valid_set_size = int(len(file_list) * args.valid_set_size)
    else:
        valid_set_size = int(args.valid_set_size)
    train_file_list = "/root/Github/TTS_Tokenizer/data/rep_wav_train.txt"
    valid_file_list = "/root/Github/TTS_Tokenizer/data/rep_wav_eval.txt"
    segment_size = cfg.get('segment_size')
    random.seed(args.split_seed)
    random.shuffle(file_list)
    print(f'A total of {len(file_list)} samples will be processed, and {valid_set_size} of them will be included in the validation set.')
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav, sr = torchaudio.load(audio_file)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.size(-1) < segment_size:
                wav = torch.nn.functional.pad(wav, (0, segment_size - wav.size(-1)), 'constant')
            wav=wav.squeeze(0)
            input_features, attention_mask=extract_features(processor,wav)
            input_features = input_features.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            vq_emb = semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = vq_emb.hidden_states[17]
            rep = (feat - semantic_mean.to(feat)) / semantic_std.to(feat)

            if str(path1) in audio_file:
                rep_file = audio_file.replace(str(path1), f"{args.rep_dir}/clean100").split('.')[0] +'.hubert.npy'
            # else :
            #     rep_file = audio_file.replace(str(path2), f"{args.rep_dir}/clean360").split('.')[0] + '.hubert.npy'
            rep_sub_dir = '/'.join(rep_file.split('/')[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            np.save(rep_file, rep.detach().cpu().numpy())
            if i < valid_set_size:
                with open(valid_file_list, 'a+') as f:
                    f.write(f'{audio_file}\t{rep_file}\n')
            else:
                with open(train_file_list, 'a+') as f:
                    f.write(f'{audio_file}\t{rep_file}\n')
            
            

