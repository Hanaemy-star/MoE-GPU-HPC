from pathlib import Path

import torch


def load_data(path: str):

    raw = Path(path).read_bytes()

    tokens = torch.tensor(
        list(raw),
        dtype=torch.long
    )

    split = int(len(tokens) * 0.9)

    train_data = tokens[:split]
    val_data = tokens[split:]

    return train_data, val_data

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    ix = torch.randint(
        low=0, 
        high=len(data) - block_size, 
        size=(batch_size,),
        )
    
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)

    return (x, y)
