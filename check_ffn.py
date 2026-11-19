import torch

from src.models.ffn import DenseFFN

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DenseFFN(
    d_model=64,
    d_ff=128
).to(device)

x = torch.randn(
    4, 16, 64,
    device=device
)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)

assert y.shape == x.shape

loss = y.square().mean()

print("Loss:", loss.item())

loss.backward()

for name, param in model.named_parameters():

    print(
        name,
        "shape:", param.shape,
        "has_grad:", param.grad is not None
    )

    assert param.grad is not None
    assert torch.isfinite(param.grad).all()

print("All checks passed!")
