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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config/spt_base_cfg.json")
    parser.add_argument('--rep_dir', type=str, default="/mnt/nfs3/zhangjinouwen/dataset/rep/rep_small_avg_hubert")
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav,mp3')
    parser.add_argument('--split_seed', type=int, help="Random seed", default=0)
    args = parser.parse_args()
    exts = args.exts.split(',')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = json.load(f)
    sample_rate = cfg.get('sample_rate')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960",cache_dir="/checkpoint")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960",cache_dir="checkpoint").eval().to(device)
    target_layer = cfg.get('semantic_model_layer')
    
    path1 = Path("/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-360")

    file_list = [
        str(file) for ext in exts 
        for path in [path1] 
        for file in path.glob(f'**/*.{ext}')
    ]
    
    train_file_list = "/mnt/nfs3/zhangjinouwen/dataset/rep/rep_middle_avg_hubert_train.txt"
    segment_size = cfg.get('segment_size')

    random.shuffle(file_list)
    print(f'A total of {len(file_list)} samples will be processed')

    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav, sr = torchaudio.load(audio_file)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.size(-1) < segment_size:
                wav = torch.nn.functional.pad(wav, (0, segment_size - wav.size(-1)), 'constant')
            input_values = feature_extractor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
            ouput = model(input_values.to(model.device), output_hidden_states=True)
            # if target_layer == 'avg':
            rep = torch.mean(torch.stack(ouput.hidden_states), axis=0)
            # else:
            #     rep = ouput.hidden_states[target_layer]

            if str(path1) in audio_file:
                rep_file = audio_file.replace(str(path1), f"{args.rep_dir}/clean360").split('.')[0] +'.hubert.npy'
            
            rep_sub_dir = '/'.join(rep_file.split('/')[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            np.save(rep_file, rep.detach().cpu().numpy())
            with open(train_file_list, 'a+') as f:
                f.write(f'{audio_file}\t{rep_file}\n')
            
            

