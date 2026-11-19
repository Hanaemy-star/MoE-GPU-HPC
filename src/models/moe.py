import torch
import torch.nn as nn
from .ffn import DenseFFN

class MoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
    ):

        super().__init__()

        self.num_experts = num_experts
        self.experts = nn.ModuleList([DenseFFN(d_model, d_ff) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_r = x.reshape(-1, x.shape[-1])
        logits = self.router(x_r)
        logits = torch.softmax(logits, dim=-1)

        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        aux_loss, counts = self.load_balancing_loss(logits, indices)
        
        outputs = torch.zeros_like(x_r)
        
        for i in range(len(self.experts)):
            token_ids, slots = torch.where(indices == i)
            if token_ids.numel() == 0:
                continue
                
            expert_input = x_r[token_ids]
            expert_weights = weights[token_ids, slots]

            expert_logits = self.experts[i](expert_input)

            expert_logits = expert_logits * expert_weights.unsqueeze(-1)

            outputs.index_add_(0, token_ids, expert_logits)

        return outputs.reshape_as(x), aux_loss, counts

    def load_balancing_loss(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        num_tokens = probs.shape[0]
        
        count = torch.bincount(indices.flatten(), minlength=self.num_experts)
        f =  count / (num_tokens * self.top_k)
        mean_probs = torch.mean(probs, dim=0)

        aux_loss = self.num_experts * (f * mean_probs).sum()

        return (aux_loss, count) 

