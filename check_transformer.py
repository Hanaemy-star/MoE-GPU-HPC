import torch
import torch.nn.functional as F

from src.models.transformer import TransformerLM

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

for use_moe in (False, True):

    model = TransformerLM(
        vocab_size=256,
        max_seq_len=128,
        d_model=64,
        d_ff=128,
        num_heads=4,
        num_layers=2,
        use_moe=use_moe,
        num_experts=4,
        top_k=2
    ).to(device)

    tokens = torch.randint(
        0, 256, (4, 33), device=device
    )

    x = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, aux_loss, counts = model(x)

    assert logits.shape == (4, 32, 256)
    assert torch.isfinite(logits).all()

    lm_loss = F.cross_entropy(
        logits.reshape(-1, 256),
        targets.reshape(-1)
    )

    loss = lm_loss + 0.01 * aux_loss
    loss.backward()

    print("\nModel:", "MoE" if use_moe else "Dense")
    print("Logits shape:", logits.shape)
    print("LM loss:", lm_loss.item())
    print("Aux loss:", aux_loss.item())
    print("Expert counts:", counts)

    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    if use_moe:
        assert counts.shape == (2, 4)
        assert (counts.sum(dim=-1) == 4 * 32 * 2).all()
        assert aux_loss.requires_grad
    else:
        assert counts is None
        assert aux_loss.item() == 0

    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()

    print("All checks passed!")
