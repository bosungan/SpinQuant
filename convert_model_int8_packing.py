"""
Convert consolidated.00.pth (qweight packed as 8 int4 per int32)
         → consolidated.01.pth (qweight packed as 2 int4 per int8)

Current int32 layout (little-endian):
  bits [3:0]=v0, [7:4]=v1, [11:8]=v2, [15:12]=v3,
  bits [19:16]=v4, [23:20]=v5, [27:24]=v6, [31:28]=v7

Each int32 byte, in memory order:
  byte0 = v0 | (v1 << 4)
  byte1 = v2 | (v3 << 4)
  byte2 = v4 | (v5 << 4)
  byte3 = v6 | (v7 << 4)

So reinterpreting the int32 tensor as int8 (view) already gives the
correct 2-int4-per-int8 packing on a little-endian system.

Shape change per qweight tensor:
  (in_features, out_features // 8)  int32
→ (in_features, out_features // 2)  int8
"""

import sys
import torch

def convert_qweight_int32_to_int8(qweight: torch.Tensor) -> torch.Tensor:
    assert qweight.dtype == torch.int32, f"Expected int32, got {qweight.dtype}"
    # Ensure contiguous so view() works without error
    qweight = qweight.contiguous()
    # Each int32 → 4 bytes → 4 int8 values, last dim expands ×4
    return qweight.view(torch.int8)


def convert(src: str, dst: str) -> None:
    print(f"Loading {src} ...")
    state_dict = torch.load(src, map_location="cpu", weights_only=False)

    converted = 0
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if key.endswith(".qweight") and tensor.dtype == torch.int32:
            old_shape = tuple(tensor.shape)
            state_dict[key] = convert_qweight_int32_to_int8(tensor)
            new_shape = tuple(state_dict[key].shape)
            print(f"  {key}: {old_shape} int32 → {new_shape} int8")
            converted += 1

    print(f"\nConverted {converted} qweight tensors.")
    print(f"Saving to {dst} ...")
    torch.save(state_dict, dst)
    print("Done.")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "consolidated.00.pth"
    dst = sys.argv[2] if len(sys.argv) > 2 else "consolidated.01.pth"
    convert(src, dst)
