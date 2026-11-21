import torch
import torch.nn as nn

from .ffn import DenseFFN
from .moe import MoE
from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2
    ):
        super().__init__()

        self.use_moe = use_moe

        self.ln1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)

        self.att = CausalSelfAttention(d_model, num_heads)

        if use_moe:
            self.ffn = MoE(d_model, d_ff, num_experts, top_k)
        else:
            self.ffn = DenseFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        H = x + self.att(self.ln1(x))

        if self.use_moe:
            Y, aux_loss, count = self.ffn(self.ln2(H))
            Y = Y + H
            return (Y, aux_loss, count)
        Y = H + self.ffn(self.ln2(H))
        return (Y, x.new_zeros(()), None) 


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k
            )
            for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)

        self.lmhead = nn.Linear(d_model, vocab_size, bias=False)

        self.num_experts = num_experts
        
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, T = tokens.shape

        embeddings = self.token_embedding(tokens)
        pos = torch.arange(T, device=tokens.device)

        x = embeddings + self.position_embedding(pos)
        
        aux_list = []
        counts_list = []
        for layer in self.layers:
            x, aux, count = layer(x)
            aux_list.append(aux)
            counts_list.append(count)

        x = self.lmhead(self.ln(x))

        aux_mean = torch.stack(aux_list).mean()

        if count is not None:
            counts = torch.stack(counts_list, dim=0)
            return (x, aux_mean, counts)
        return (x, aux_mean, None)
