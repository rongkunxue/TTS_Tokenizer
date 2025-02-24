from huggingface_hub import hf_hub_download
import torch

from moshi.models import loaders, LMGen


# of the frame size, so that you always know how many time steps you get back in `codes`.

# Now if you have a GPU around.
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device='cpu')
mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.

wav = torch.randn(1, 1, 24000 )  # should be [B, C=1, T]
with torch.no_grad():
    codes = mimi.encode(wav)  # [B, K = 8, T]
    decoded = mimi.decode(codes)