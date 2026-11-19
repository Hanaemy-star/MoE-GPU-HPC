import torch

from src.models.ffn import DenseFFN

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DenseFFN(
    d_model=64,
    d_ff=128
).to(device).double()

x = torch.randn(
    4, 16, 64,
    device=device,
    dtype=torch.float64
)

# Эталонный backward через PyTorch Autograd

y = model(x)
loss = y.square().mean()
loss.backward()


# Самостоятельная реализация forward и backward

with torch.no_grad():

    X = x.reshape(-1, 64)

    W1 = model.ffn1.weight
    b1 = model.ffn1.bias

    W2 = model.ffn2.weight
    b2 = model.ffn2.bias

    # TODO: вычислить Z, H, Y
    Z = X @ W1.T + b1
    H = Z * torch.sigmoid(Z)
    Y = H @ W2.T + b2
    # TODO: вычислить производную Loss по Y
    G_y = 2 * Y / torch.numel(Y)
    # TODO: вычислить градиенты W2 и b2
    G_w2 = G_y.T @ H
    G_b2 = G_y.sum(dim=0)
    # TODO: передать градиент через второй слой
    G_h = G_y @ W2
    # TODO: вычислить производную SiLU
    G_silu = torch.sigmoid(Z) + Z * torch.sigmoid(Z) * (1 - torch.sigmoid(Z))
    # TODO: вычислить градиенты W1 и b1
    G_z = G_h * G_silu
    G_w1 = G_z.T @ X
    G_b1 = G_z.sum(dim=0)
    G_x = G_z @ W1

    torch.testing.assert_close(G_w1, model.ffn1.weight.grad)
    torch.testing.assert_close(G_b1, model.ffn1.bias.grad)

    torch.testing.assert_close(G_w2, model.ffn2.weight.grad)
    torch.testing.assert_close(G_b2, model.ffn2.bias.grad)

    print("All manual gradients match Autograd!")
