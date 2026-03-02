import torch
from torch import nn
from torch.nn import functional as F
import math

# Self Attention은 각 토큰이 다른 토큰들을 얼마나 참고해서 업데이트 할지를 결정
class SelfAttention(nn.Module):

    # Self-Attention은 Query, Key, Vector 를 이용해 계산을 해야한다.
    # Q = X * WQ = "내가 찾고 싶은 것, 즉 현재 토큰이 어떤 정보를 원하는 지 나타내는 벡터"
    # K = X * WK = "각 토큰이 어떤 성질을 갖고 있는 지 나타내는 벡터"
    # V = X * WV = "실제로 전달할 데이터"

    # n_heads 는 attention head 개수인데, "헤드"는 attention을 여러 개 병렬로 돌리는 통로이다.
    # 여러 head를 두면, head1 = "가까운 문맥 관계" head2 ="먼 문맥 관계" head3 = "특정 패턴 역할"
    # 이처럼 다양한 관점의 관계를 파악할 수 있다. 

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        # 한번에 Q, K, V 만들기 위해서 in_proj는 3 * d_embed의 차원 수를 가짐
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # 헤드 개수당 차원수

    def forward(self, x: torch.Tensor, causal_mask = False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Q, K, V가 proj를 거치면 [Batch_Size, Token, 차원 수] 이렇게 되는데
        # 멀티 헤드를 하려면 차원 수를 d_head(헤드개수당 차원수)로 바꿔야함 
        # 따라서 이렇게 shape을 변경
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # 우선 q와 k의 내적을 통해서 유사도를 계산
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # i번째 토큰에서 자기보다 뒤에 있는 토큰(i+1부터)을 True로 싹 배치
            weight.masked_fill_(mask, -torch.inf)
            # -무한대 이므로, softmax에서 확률이 0. 즉 미래를 못보도록 masking

        # 위에서 q와 k를 내적했는데, 차원이 커질수록 내적값은 자연스럽게 커진다.
        # 커지면 softmax함수에 큰값들이 들어가는데, exp에 큰제곱값들이 들어가면서 그 값이 매우 커지게 된다.
        # 따라서 이렇게 d_head(차원의 개수)로 나눠서 스케일을 고려한다.
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)

        # softmax까지 마친 그 비율로 v를 섞는다.
        output = weight @ v
        output = output.transpose(1,2)
        output = output.reshape(input_shape)
        output = self.out_proj(output) # Linear에 넣고 마지막으로 한번 더 섞기

        return output

# Cross Attention은 서로 다른 입력 소스간의 관계를 모델링하여 문맥을 강화한다.
# 예를 들어, 이미지가 텍스트를 참고하여 이미지를 업데이트 하는 것이 Cross-Attention이다.
class CrossAttention(nn.Module):
    # d_embed = x 토큰 vector 길이, d_cross = y 토큰 벡터 길이
    # q는 x에서 만드므로 업데이트할 대상
    # k와 v는 y에서 만드므로 참고할 대상
    def __init__(self, n_heads: int, d_embed:int, d_cross:int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads # head당 차원 수

    # x = 업데이트할 대상, y = 참고할 대상
    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        intermim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # 아래 유사도 계산 part는 앞선 SelfAttention과 똑같다.
        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim = -1)

        output = weight @ v
        output = output.transpose(1,2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output