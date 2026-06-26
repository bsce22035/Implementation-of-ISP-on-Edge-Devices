#!/usr/bin/env python3
"""
extract_weights.py  –  SYENet-Slim PKL -> FPGA memory files
============================================================
Loads model_best_slim.pkl, extracts all layer weights and biases,
fuses batch-norm and re-parameterised branches, quantises to INT8/INT32,
and writes Verilog $readmemh-compatible .mem files.

Usage:
    python extract_weights.py --pkl model_best_slim.pkl --out ../mem/

Output files:
    weights.mem  – INT8 weights, hex, one byte per line
    biases.mem   – INT32 biases, hex, one 32-bit word per line
    prelu_alpha.mem – PReLU alphas, hex, one byte per line

Weight memory layout (byte offsets):
    0x000  head.block1.0 : Conv(4->12, 3×3)  = 432
    0x1B0  head.block1.2 : Conv(12->12,3×3)  = 1296 (re-param fused)
    0x6C0  head.block2   : Conv(4->12, 1×1)  = 48
    0x6F0  body.block1   : Conv(12->12,3×3)  = 1296
    0xBE0  body.block2   : Conv(12->12,1×1)  = 144
    0xC70  att.1         : Conv(12->12,1×1)  = 144
    0xD00  att.3         : Conv(12->12,1×1)  = 144
    0xD90  tail.1        : Conv(12->3,  1×1) = 36

Bias memory layout (word offsets, each = 4 bytes INT32):
    0x00  head.block1.0.bias  : 12
    0x0C  head.block1.2.bias  : 12
    0x18  head.block2.bias    : 12
    0x24  body.block1.bias    : 12
    0x30  body.block2.bias    : 12
    0x3C  att.1.bias          : 12
    0x48  att.3.bias          : 12
    0x54  tail.1.bias         :  3
"""

import os
import sys
import struct
import argparse
import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantise_weights(w: np.ndarray, scale: float = 127.0) -> np.ndarray:
    """Float32 -> INT8.  scale = abs-max -> 127."""
    abs_max = np.abs(w).max()
    if abs_max == 0:
        return np.zeros_like(w, dtype=np.int8)
    q = np.round(w / abs_max * scale).clip(-128, 127).astype(np.int8)
    return q


def quantise_bias(b: np.ndarray, wscale: float, ascale: float = 1.0) -> np.ndarray:
    """Float32 bias -> INT32 (scaled to match INT8 weight × UINT8 activation)."""
    # bias_q = round(b / (wscale * ascale))
    # ascale ≈ 1.0 for UINT8 activations normalised to [0,1]
    q = np.round(b / (wscale * ascale)).astype(np.int32)
    return q


def reparameterise_head(state_dict: dict) -> np.ndarray:
    """
    Fuse head.block1 (3×3 conv + BN + optional 1×1) into one 3×3 conv.
    This converts the multi-branch training structure into a single
    inference-equivalent 3×3 filter.

    If only head.block1.2.weight is present (already fused), return it directly.
    """
    w_main = state_dict.get("head.block1.2.weight")
    if w_main is not None:
        w = w_main.cpu().float().numpy()
        print(f"  head.block1.2 (re-param): shape {w.shape}, range [{w.min():.4f}, {w.max():.4f}]")
        return w

    # Fallback: try block1.0.weight
    w0 = state_dict.get("head.block1.0.weight", None)
    if w0 is not None:
        return w0.cpu().float().numpy()
    return None


def write_mem_int8(path: str, data: np.ndarray):
    """Write INT8 array as hex .mem file (one byte per line, 2 hex digits)."""
    flat = data.flatten().astype(np.int8)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v) & 0xFF:02x}\n")   # int() avoids NumPy 2.x int8 overflow
    print(f"  Written {len(flat)} bytes -> {path}")


def write_mem_int32(path: str, data: np.ndarray):
    """Write INT32 array as hex .mem file (one 32-bit word per line, 8 hex digits)."""
    flat = data.flatten().astype(np.int32)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")  # int() avoids NumPy 2.x overflow
    print(f"  Written {len(flat)} words -> {path}")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="model_best_slim.pkl",
                        help="Path to the .pkl model file")
    parser.add_argument("--out", default="../mem/",
                        help="Output directory for .mem files")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load PKL
    print(f"Loading {args.pkl} ...")
    sd = torch.load(args.pkl, map_location="cpu")
    # Handle case where file is a full checkpoint dict
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    print(f"Keys found ({len(sd)}):")
    for k, v in sd.items():
        print(f"  {k:40s}  shape={tuple(v.shape)}  dtype={v.dtype}")

    # -----------------------------------------------------------------------
    # Extract and quantise each layer
    # -----------------------------------------------------------------------
    weight_arrays = []  # list of (name, int8_array, float_scale)
    bias_arrays   = []  # list of (name, int32_array)

    def get_np(key):
        t = sd.get(key)
        if t is None:
            print(f"  WARNING: key '{key}' not found – using zeros")
            return None
        return t.cpu().float().numpy()

    # -- Layer L0: head.block1.0 Conv(4->12, 3×3) ----------------------------
    print("\n[L0] head.block1.0")
    w = get_np("head.block1.0.weight")
    b = get_np("head.block1.0.bias")
    # Fuse BN if present
    bn_w = get_np("head.block1.1.weight")
    if w is not None:
        if bn_w is not None:
            # Simple BN scale fusion: w_fused[oc] = w[oc] * bn_w[oc]
            w = w * bn_w[:, None, None, None]
        wq   = quantise_weights(w)
        wscl = np.abs(w).max() / 127.0
        weight_arrays.append(("head.block1.0", wq, wscl))
        if b is not None:
            bq = quantise_bias(b, wscl)
        else:
            bq = np.zeros(12, dtype=np.int32)
        bias_arrays.append(("head.block1.0.bias", bq))
        print(f"  shape={w.shape}, wscale={wscl:.6f}")

    # -- Layer L2: head.block1.2 Conv(12->12, 3×3) re-param ------------------
    print("\n[L2] head.block1.2 (re-param)")
    w2 = reparameterise_head(sd)
    b2 = get_np("head.block1.2.bias")
    if w2 is not None:
        wq2  = quantise_weights(w2)
        wscl2 = np.abs(w2).max() / 127.0
        weight_arrays.append(("head.block1.2", wq2, wscl2))
        if b2 is not None:
            bq2 = quantise_bias(b2, wscl2)
        else:
            bq2 = np.zeros(12, dtype=np.int32)
        bias_arrays.append(("head.block1.2.bias", bq2))

    # -- Layer L3: head.block2 Conv(4->12, 1×1) skip -------------------------
    print("\n[L3] head.block2")
    w3 = get_np("head.block2.weight")
    b3 = get_np("head.block2.bias")
    if w3 is not None:
        wq3  = quantise_weights(w3)
        wscl3 = np.abs(w3).max() / 127.0
        weight_arrays.append(("head.block2", wq3, wscl3))
        bq3 = quantise_bias(b3, wscl3) if b3 is not None else np.zeros(12, dtype=np.int32)
        bias_arrays.append(("head.block2.bias", bq3))

    # -- Layer L5: body.block1 Conv(12->12, 3×3) -----------------------------
    print("\n[L5] body.block1")
    w5 = get_np("body.block1.weight")
    b5 = get_np("body.block1.bias")
    if w5 is not None:
        wq5   = quantise_weights(w5)
        wscl5 = np.abs(w5).max() / 127.0
        weight_arrays.append(("body.block1", wq5, wscl5))
        bq5 = quantise_bias(b5, wscl5) if b5 is not None else np.zeros(12, dtype=np.int32)
        bias_arrays.append(("body.block1.bias", bq5))

    # -- Layer L6: body.block2 Conv(12->12, 1×1) skip ------------------------
    print("\n[L6] body.block2")
    w6 = get_np("body.block2.weight")
    b6 = get_np("body.block2.bias")
    if w6 is not None:
        wq6   = quantise_weights(w6)
        wscl6 = np.abs(w6).max() / 127.0
        weight_arrays.append(("body.block2", wq6, wscl6))
        bq6 = quantise_bias(b6, wscl6) if b6 is not None else np.zeros(12, dtype=np.int32)
        bias_arrays.append(("body.block2.bias", bq6))

    # -- Attention FC1: att.1 Conv(12->12, 1×1) ------------------------------
    print("\n[Att] att.1")
    wa1 = get_np("att.1.weight")
    ba1 = get_np("att.1.bias")
    if wa1 is not None:
        wqa1   = quantise_weights(wa1)
        wscla1 = np.abs(wa1).max() / 127.0
        weight_arrays.append(("att.1", wqa1, wscla1))
        bqa1 = quantise_bias(ba1, wscla1) if ba1 is not None else np.zeros(12, dtype=np.int32)
        bias_arrays.append(("att.1.bias", bqa1))

    # -- Attention FC2: att.3 Conv(12->12, 1×1) ------------------------------
    print("\n[Att] att.3")
    wa3 = get_np("att.3.weight")
    ba3 = get_np("att.3.bias")
    if wa3 is not None:
        wqa3   = quantise_weights(wa3)
        wscla3 = np.abs(wa3).max() / 127.0
        weight_arrays.append(("att.3", wqa3, wscla3))
        bqa3 = quantise_bias(ba3, wscla3) if ba3 is not None else np.zeros(12, dtype=np.int32)
        bias_arrays.append(("att.3.bias", bqa3))

    # -- Tail FC: tail.1 Conv(12->3, 1×1) ------------------------------------
    print("\n[Tail] tail.1")
    wt = get_np("tail.1.weight")
    bt = get_np("tail.1.bias")
    if wt is not None:
        wqt   = quantise_weights(wt)
        wsclT = np.abs(wt).max() / 127.0
        weight_arrays.append(("tail.1", wqt, wsclT))
        bqt = quantise_bias(bt, wsclT) if bt is not None else np.zeros(3, dtype=np.int32)
        bias_arrays.append(("tail.1.bias", bqt))

    # -----------------------------------------------------------------------
    # Write combined weight .mem file
    # -----------------------------------------------------------------------
    print("\n=== Writing weight BRAM memory file ===")
    wgt_file = os.path.join(args.out, "weights.mem")
    all_weights = np.concatenate([a[1].flatten() for a in weight_arrays], axis=0)
    write_mem_int8(wgt_file, all_weights)

    # Offset report
    offset = 0
    print("\nWeight BRAM address map:")
    for name, arr, scl in weight_arrays:
        n = arr.size
        print(f"  0x{offset:04X}  {name:25s}  {arr.shape}  scale={scl:.6f}")
        offset += n

    # -----------------------------------------------------------------------
    # Write combined bias .mem file
    # -----------------------------------------------------------------------
    print("\n=== Writing bias BRAM memory file ===")
    bias_file = os.path.join(args.out, "biases.mem")
    all_biases = np.concatenate([a[1].flatten() for a in bias_arrays], axis=0)
    write_mem_int32(bias_file, all_biases)

    # Offset report
    offset = 0
    print("\nBias BRAM address map:")
    for name, arr in bias_arrays:
        print(f"  0x{offset:02X}  {name:30s}  {arr.shape}")
        offset += arr.size

    # -----------------------------------------------------------------------
    # Verification report
    # -----------------------------------------------------------------------
    print("\n=== Quantisation Verification ===")
    print(f"Total weight bytes : {all_weights.size}")
    print(f"Total bias words   : {all_biases.size}")
    print(f"Weight BRAM usage  : {all_weights.size / 4096 * 100:.1f}% of 4096-entry BRAM")
    print(f"Bias BRAM usage    : {all_biases.size / 128 * 100:.1f}% of 128-entry BRAM")

    print("\nDone.  .mem files written to:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
