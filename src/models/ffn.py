import torch
import torch.nn as nn

class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.silu(x)
        x = self.ffn2(x)

        return x
        
