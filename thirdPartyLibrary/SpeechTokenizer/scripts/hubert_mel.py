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
    parser.add_argument('--valid_set_size', type=float, default=1500)
    args = parser.parse_args()
    exts = args.exts.split(',')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = json.load(f)
    sample_rate = cfg.get('sample_rate')
    print(sample_rate)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960",cache_dir="/root/Github/TTS_Tokenizer/thirdPartyLibrary/SpeechTokenizer/checkpoint")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960",cache_dir="/root/Github/TTS_Tokenizer/thirdPartyLibrary/SpeechTokenizer/checkpoint").eval().to(device)
    target_layer = "avg"
    
    path2 = Path("/mnt/nfs3/zhangjinouwen/dataset/LibriTTS/train-clean-100")

    file_list = [
        str(file) for ext in exts 
        for path in [path2] 
        for file in path.glob(f'**/*.{ext}')
    ]
    train_file_list = "/mnt/nfs3/zhangjinouwen/dataset/rep/rep_small_mel_hubert_train.txt"

    segment_size = 65536
    random.seed(args.split_seed)
    random.shuffle(file_list)
    print(f'A total of {len(file_list)} samples will be processed.')
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav_24k, sr = torchaudio.load(audio_file)

            if wav_24k.size(-1) < segment_size:
                wav_24k = torch.nn.functional.pad(wav_24k, (0, segment_size - wav_24k.size(-1)), 'constant')
                wav_16k = torchaudio.functional.resample(wav_24k, sr, 16000)
            else:
                wav_16k = torchaudio.functional.resample(wav_24k, sr, 16000)
            input_values = feature_extractor(wav_16k.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values
            ouput = model(input_values.to(model.device), output_hidden_states=True)
            rep = torch.mean(torch.stack(ouput.hidden_states), axis=0)
           

            if str(path2) in audio_file:
                rep_file = audio_file.replace(str(path2), f"{args.rep_dir}/cleanmel100").split('.')[0] +'.hubert.npy'
            
            rep_sub_dir = '/'.join(rep_file.split('/')[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            np.save(rep_file, rep.detach().cpu().numpy())
            with open(train_file_list, 'a+') as f:
                f.write(f'{audio_file}\t{rep_file}\n')
            
            

