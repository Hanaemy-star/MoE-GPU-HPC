import torch

from src.models.attention import CausalSelfAttention

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CausalSelfAttention(
    d_model=64,
    num_heads=4
).to(device)

x = torch.randn(3, 5, 64, device=device)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)

assert y.shape == x.shape
assert torch.isfinite(y).all()

loss = y.square().mean()
loss.backward()

for name, param in model.named_parameters():
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()

    print(name, "grad norm:", param.grad.norm().item())

# Проверяем, что модель не смотрит в будущее

model.eval()

with torch.no_grad():
    x_modified = x.clone()
    x_modified[:, 3:, :] += 100

    y_original = model(x)
    y_modified = model(x_modified)

    torch.testing.assert_close(
        y_original[:, :3, :],
        y_modified[:, :3, :],
        atol=1e-5,
        rtol=1e-5
    )

print("Causal masking check passed!")
print("All attention checks passed!")
