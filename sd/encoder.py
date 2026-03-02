import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super(VAE_Encoder, self).__init__(
            # (Batch_Size, 3(Channel), Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            # ResidualBlock -> 표현력 증가 시키는 기능 -> 여러개 넣어서 더 표현력 up
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            # stride가 2 -> 한칸씩 건너뛰면서 CNN -> 해상도 절반
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            # Attention은 먼 거리 픽셀 간 관계를 잘 연결한다.
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), # 정규화

            nn.SiLU(),

            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            # 512채널짜리 feature map을 8채널짜리로 줄이는 Conv2d -> latent representation의 mean과 log_variance를 만들기 위해 채널 수를 8로 줄임
            nn.Conv2d(512, 8, kernel_size = 3, padding = 1),

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    # VAE의 encoder를 순서대로 통과 시키고, 마지막에 나온 output_channel을 forward에 입력
    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        for module in self:
            # module에 stride 속성이 있으면 값을 가져오고 아니면 None
            # stride가 2인 layer는 해상도를 줄이는 Conv2d인데, 이 레이어를 만나면, padding을 넣어서 해상도를 유지.
            if getattr(module, 'stride', None) == (2, 2): 
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0,1,0,1))
            x = module(x) # padding 추가한 모듈 적용

        # x를 2개의 텐서로 나누는데, mean과 log_variance로 나눈다.
        # VAE는 한 latent를 생성하는 게 아니라, latent의 확률 분포를 만들어야 하므로, 정규분포를 위해 mean, variance가 필요하다.
        # [B, 8, H/8, W/8] -> [B, 4, H/8, W/8]
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # clamp 함수는 min값보다 작은 값은 모두 min으로, max값보다 큰 값은 모두 max로
        log_variance = torch.clamp(log_variance, -30, 20)

        # logvariance -> variance
        variance = log_variance.exp()

        stdev = variance.sqrt()

        # 평균, 표준편차로 정규분포에서 sample 하나 뽑기
        x = mean + stdev * noise

        # Scale the output by a constant
        x *= 0.18215

        return x

