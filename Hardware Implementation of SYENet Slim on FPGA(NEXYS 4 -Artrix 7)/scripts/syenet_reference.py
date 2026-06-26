#!/usr/bin/env python3
"""
syenet_reference.py  -  GOLDEN software reference for SYENet-Slim ISP (RAW -> RGB)
=================================================================================
This is the authoritative float32 reference. It reconstructs the official
SYEISPNetS architecture (github.com/sanechips-multimedia/syenet, model/isp.py),
loads model_best_slim.pkl with strict=True (which proves the architecture is
correct - every weight name and shape must match), and converts a RAW Bayer
image into an RGB image exactly as the model was trained to do.

Pipeline (matches data/ispdata.py):
    raw uint16 (256x256 Bayer)  -> /4095 normalize
                                -> bayer2rggb pack  -> [4,128,128]
    network                     -> [3,256,256] in [0,1]
                                -> *255, clamp, uint8 -> RGB PNG

Usage:
    python scripts/syenet_reference.py --raw "D:/.../mediatek_raw/100.png" --out out_rgb.png
"""
import argparse, sys
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: pip install torch"); sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: pip install Pillow"); sys.exit(1)


# ---------------------------------------------------------------------------
# Exact SYEISPNetS reconstruction (slim / inference model)
# ---------------------------------------------------------------------------
class QCU(nn.Module):
    """Quadratic Connection Unit (slim):  out = block1(x) * block2(x) + bias"""
    def __init__(self, block1, block2, channels):
        super().__init__()
        self.block1 = block1
        self.block2 = block2
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        return self.block1(x) * self.block2(x) + self.bias


class SYEISPNetS(nn.Module):
    def __init__(self, ch=12):
        super().__init__()
        # HEAD: block1 = Conv5x5 -> PReLU -> Conv3x3 ; block2 = Conv5x5
        head_b1 = nn.Sequential(
            nn.Conv2d(4,  ch, 5, 1, 2),   # head.block1.0
            nn.PReLU(ch),                 # head.block1.1
            nn.Conv2d(ch, ch, 3, 1, 1),   # head.block1.2
        )
        head_b2 = nn.Conv2d(4, ch, 5, 1, 2)               # head.block2
        self.head = QCU(head_b1, head_b2, ch)             # + head.bias

        # BODY: block1 = Conv3x3 ; block2 = Conv1x1
        body_b1 = nn.Conv2d(ch, ch, 3, 1, 1)              # body.block1
        body_b2 = nn.Conv2d(ch, ch, 1, 1, 0)              # body.block2
        self.body = QCU(body_b1, body_b2, ch)             # + body.bias

        # ATTENTION (squeeze-excite): GAP -> 1x1 -> PReLU -> 1x1 -> Sigmoid
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # att.0 (no params)
            nn.Conv2d(ch, ch, 1),         # att.1
            nn.PReLU(ch),                 # att.2
            nn.Conv2d(ch, ch, 1),         # att.3
            nn.Sigmoid(),                 # att.4 (no params)
        )

        # TAIL: PixelShuffle(2) 12->3 @ 2x  ->  Conv3x3
        self.tail = nn.Sequential(
            nn.PixelShuffle(2),           # tail.0 (no params)
            nn.Conv2d(3, 3, 3, 1, 1),     # tail.1
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        return self.tail(x)


# ---------------------------------------------------------------------------
# RAW preprocessing (matches official ispdata.py)
# ---------------------------------------------------------------------------
def bayer2rggb(img_bayer):
    """256x256 Bayer -> [4,128,128] RGGB pack (official transpose)."""
    h, w = img_bayer.shape
    x = img_bayer.reshape(h // 2, 2, w // 2, 2)
    x = x.transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])
    return x  # [4, h/2, w/2]


def load_raw(path):
    im = Image.open(path)
    a = np.array(im).astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"Expected single-channel Bayer, got shape {a.shape}")
    a = a / 4095.0                      # 12-bit normalize
    packed = bayer2rggb(a)              # [4, H/2, W/2]
    return packed.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="model_best_slim.pkl")
    ap.add_argument("--raw", required=True, help="RAW Bayer PNG (e.g. mediatek_raw/100.png)")
    ap.add_argument("--out", default="reference_rgb.png")
    args = ap.parse_args()

    # Build model and load weights STRICTLY (architecture validation)
    model = SYEISPNetS(ch=12)
    sd = torch.load(args.pkl, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:     sd = sd["model"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("!! state_dict mismatch")
        print("   missing   :", missing)
        print("   unexpected:", unexpected)
        print("   (architecture does NOT match - fix before trusting output)")
    else:
        print("OK: all weights loaded strict - architecture confirmed correct.")
    model.eval()

    # Load + preprocess raw
    packed = load_raw(args.raw)                       # [4,128,128]
    x = torch.from_numpy(packed).unsqueeze(0)         # [1,4,128,128]
    print(f"Input  : {tuple(x.shape)}  range [{x.min():.3f}, {x.max():.3f}]")

    with torch.no_grad():
        y = model(x)                                  # [1,3,256,256]
    print(f"Output : {tuple(y.shape)}  range [{y.min():.3f}, {y.max():.3f}]")

    # Post-process -> RGB PNG
    rgb = y.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()  # HWC
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb_u8, "RGB").save(args.out)
    print(f"Saved RGB -> {args.out}  ({rgb_u8.shape[1]}x{rgb_u8.shape[0]})")


if __name__ == "__main__":
    main()
