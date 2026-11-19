import torch

from src.models.moe import MoE

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MoE(
    d_model=64,
    d_ff=128,
    num_experts=4,
    top_k=2
).to(device)

x = torch.randn(3, 5, 64, device=device)

y, _, _  = model(x)

print("Device:", device)
print("Input shape:", x.shape)
print("Output shape:", y.shape)

assert y.shape == x.shape
assert torch.isfinite(y).all()

loss = y.square().mean()
loss.backward()

print("Loss:", loss.item())

for name, param in model.named_parameters():
    if param.grad is not None:
        assert torch.isfinite(param.grad).all()
        print(name, "grad norm:", param.grad.norm().item())
    else:
        print(name, "no gradient")

assert model.router.weight.grad is not None
assert model.router.weight.grad.norm() > 0

print("All MoE checks passed!")
