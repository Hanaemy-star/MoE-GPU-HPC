import torch

from src.data import load_data, get_batch


train_data, val_data = load_data(
    "data/tiny_shakespeare.txt"
)

print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))

assert len(train_data) > 0
assert len(val_data) > 0

assert train_data.dtype == torch.long
assert val_data.dtype == torch.long

assert train_data.min() >= 0
assert train_data.max() < 256


device = "cuda" if torch.cuda.is_available() else "cpu"

x, y = get_batch(
    data=train_data,
    batch_size=8,
    block_size=64,
    device=device
)

print("X shape:", x.shape)
print("Y shape:", y.shape)

print("X device:", x.device)
print("Y device:", y.device)

assert x.shape == (8, 64)
assert y.shape == (8, 64)

assert x.dtype == torch.long
assert y.dtype == torch.long

assert x.device.type == device
assert y.device.type == device

# Проверяем сдвиг последовательности

assert torch.equal(
    x[:, 1:],
    y[:, :-1]
)

print("All data checks passed!")
