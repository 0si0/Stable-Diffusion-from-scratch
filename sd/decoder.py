import torch
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        n, c, h, w = x.shape

        x = self.groupnorm(x)
        
        # view = 텐서의 shape을 바꾸는 연산
        x = x.view(n, c, h * w) 

        x = x.transpose(-1, -2) # 마지막 두 차원 바꿈 [n, c, h*w] -> [n, h*w, c]

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)

        x += residue # 원본 정보 유지

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Residual Block은 변환을 2번 하는데, 첫 변환은 in_channels -> out_channel, 둘째변환은 out_channels-> out_channels
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # conv 하나로는 표현이 얕아서 2개로 더 복잡하게 진행
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.groupnorm_1(x) # feature의 스케일/분포가 뒤죽박죽 일 수 있어서 항상 정규화 먼저
        x = F.silu(x) # 비선형함수 SiLu 활성화
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        # 만약 이 블록이 in = 128 out = 256이면 
        # residue랑 x랑 채널이 달라서 덧셈이 불가능 -> 따라서 residual_layer 적용 후 더함
        # residue를 더하는 이유는 이 블록이 원래 정보를 "수정"하는 역할이기 때문. 따라서 원본을 보존
        return x + self.residual_layer(residue)
    

# Encoder랑 반대로
class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # padding = 0 이면 해상도가 줄어들기 때문에, decoder에서는 패딩을 1로 해서 해상도 유지
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            nn.Upsample(scale_factor=2), 

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),   

            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),  

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215

        for module in self:
            x = module(x)
        
        # (Batch_Size, 3, H, W)
        return x
    