import torch
from vector_quantize_pytorch import SimVQ
import torch.nn as nn

sim_vq = SimVQ(
    dim = 512,
    codebook_size = 8192,
    rotation_trick = True,  # use rotation trick from Fifty et al.
    codebook_transform = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ),
    channel_first= True
)

x = torch.randn(10, 1024,50)
quantized, indices, commit_loss = sim_vq(x)

assert x.shape == quantized.shape
assert torch.allclose(quantized, sim_vq.indices_to_codes(indices), atol = 1e-6)