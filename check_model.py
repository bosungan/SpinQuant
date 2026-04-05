import torch

ckpt = torch.load("consolidated.00.pth", map_location="cpu", weights_only=False)

def walk(obj, prefix=""):
    if torch.is_tensor(obj):
        print(f"{prefix} -> TENSOR shape={tuple(obj.shape)} dtype={obj.dtype}")

    elif isinstance(obj, dict):
        print(f"{prefix} -> dict ({len(obj)} keys)")
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            walk(v, new_prefix)

    elif isinstance(obj, list):
        print(f"{prefix} -> list (len={len(obj)})")
        for i, v in enumerate(obj):
            walk(v, f"{prefix}[{i}]")

    elif isinstance(obj, tuple):
        print(f"{prefix} -> tuple (len={len(obj)})")
        for i, v in enumerate(obj):
            walk(v, f"{prefix}[{i}]")

    else:
        # 핵심: tensor 말고도 다 출력
        print(f"{prefix} -> {type(obj)} : {obj}")

walk(ckpt)