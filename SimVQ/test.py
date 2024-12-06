import torch
import torch.nn as nn

class VideoConv(nn.Module):
    def __init__(self, input_channels):
        super(VideoConv, self).__init__()
        self.transconv = nn.ConvTranspose1d(
            in_channels=input_channels, 
            out_channels=input_channels,  # 通道数不变
            kernel_size=6,                # 卷积核大小
            stride=8,  
            padding=2,
            output_padding=1,
        )
    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.transconv(x) 
        x = x.transpose(1, 2) 
        return x

# 输入形状 (batch_size, channels, sequence_length)
input_tensor = torch.randn(8, 10, 64)  # (8, 64, 10)
model = VideoConv(input_channels=64)
output_tensor = model(input_tensor)

print("Output shape:", output_tensor.shape)  
