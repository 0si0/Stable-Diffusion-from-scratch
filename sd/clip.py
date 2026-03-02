import torch
from torch import nn
from torch.nn import functional as f
from sd.attention import SelfAttention

# ** CLIP의 3가지 고정 숫자 알아두기 **
# n_vocab = 단어장 크기 49408, 가능한 토큰 ID 개수
# n_embed = 임베딩 차원 768, 토큰 하나를 768개의 숫자로 표현
# n_tokens = 최대 토큰 길이, 문장 길이 제한 


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):

        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x

# Attention 1번 -> FeddForward 1번 하는 블록
class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        residue = x

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask = True)

        x += residue

        residue = x

        x = self.layernorm_2(x)

        # Linear를 한번만 사용하면 선형 변환이라서 복잡한 형태의 변환을 하지 못한다. 
        # 따라서 Linear를 2개 활용하고 사이에 비선형 활성화 함수를 넣어서 복잡한 함수를 만들어 낸다.
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # activation function

        x = self.linear_2(x)

        x += residue

        return x



class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module(
            [CLIPLayer(12, 768) for i in range(12)]
        )

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output