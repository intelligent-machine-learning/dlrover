import torch
import torch.nn.functional as F
from torch import nn

from atorch.modules.distributed_modules.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

from .model_args import ModelArgs


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        # hidden_dim = int(2 * hidden_dim / 3)
        # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=True)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=True, input_is_parallel=True)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=True)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size,
            params.dim,  # init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(FeedForward(dim=params.dim, hidden_dim=4 * params.dim, multiple_of=params.multiple_of))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = torch.nn.Linear(
            params.dim,
            params.dim,
            bias=False,  # init_method=lambda x: x
        )

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # print(f'h: {h}, freqs_cis: {freqs_cis}, mask: {mask}')
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.norm(h)
        output = self.output(h)
        return output
