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

x = torch.randn(8, 128, 64, device=device)

y, aux_loss, counts = model(x)

main_loss = y.square().mean()
loss = main_loss + 0.01 * aux_loss

loss.backward()

print("Output shape:", y.shape)
print("Main loss:", main_loss.item())
print("Auxiliary loss:", aux_loss.item())
print("Total loss:", loss.item())
print("Expert counts:", counts)
print("Expert distribution:", counts / counts.sum())

assert y.shape == x.shape
assert counts.shape == (4,)
assert counts.sum().item() == 8 * 128 * 2
assert aux_loss.ndim == 0
assert torch.isfinite(aux_loss)
assert aux_loss.requires_grad
assert model.router.weight.grad is not None
assert torch.isfinite(model.router.weight.grad).all()

# Проверка математической корректности
# на равномерном распределении

probs = torch.full((4, 4), 0.25, device=device)

indices = torch.tensor([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]
], device=device)

balanced_loss, balanced_counts = (
    model.load_balancing_loss(probs, indices)
)

torch.testing.assert_close(
    balanced_loss,
    torch.tensor(1.0, device=device)
)

print("All load balancing checks passed!")
