import sys, os, numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), '..')
SIM = os.path.join(ROOT, 'sim')


def rd(path):
    v = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                v.append(int(ln, 16))
    a = np.array(v, dtype=np.int32)
    a = np.where(a >= 128, a - 256, a)   # int8 two's complement
    return a


def rdu(path):
    v = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                v.append(int(ln, 16))
    return np.array(v, dtype=np.uint8)


tag = sys.argv[1] if len(sys.argv) > 1 else 'att'

# gv
g_gv = rd(os.path.join(SIM, 'g_gv.hex'))
r_gv = rd(os.path.join(SIM, f'rtl_{tag}_gv.mem'))
print("golden gv:", g_gv.tolist())
print("rtl    gv:", r_gv.tolist())
print("gv match:", np.array_equal(g_gv, r_gv))

# att (M0)
g_att = rd(os.path.join(SIM, 'g_att.hex'))
r_att = rd(os.path.join(SIM, f'rtl_{tag}_m0.mem'))
n = min(len(g_att), len(r_att))
eq = np.array_equal(g_att[:n], r_att[:n])
print(f"\natt (M0) match: {eq}   lens g={len(g_att)} r={len(r_att)}")
if not eq:
    d = np.where(g_att[:n] != r_att[:n])[0]
    print(f"  mismatches: {len(d)}/{n}  first idx {d[:8].tolist()}")
    for idx in d[:5]:
        c = idx // 16384; p = idx % 16384; y = p // 128; x = p % 128
        print(f"   idx {idx} (c={c},y={y},x={x}) golden={g_att[idx]} rtl={r_att[idx]}")
    # per-channel mismatch counts
    perc = [int(np.sum(g_att[c*16384:(c+1)*16384] != r_att[c*16384:(c+1)*16384])) for c in range(12)]
    print("  per-channel mismatch counts:", perc)

# rgb vs reference png
r_rgb = rdu(os.path.join(SIM, f'rtl_{tag}_rgb.mem'))
ref = np.array(Image.open(os.path.join(ROOT, 'int8_integer_out.png')).convert('RGB')).astype(np.int64)
if len(r_rgb) == 196608:
    rgb = r_rgb.reshape(256, 256, 3).astype(np.int64)
    mse = np.mean((rgb - ref) ** 2)
    psnr = 99.0 if mse == 0 else 10 * np.log10(255**2 / mse)
    print(f"\nRTL rgb vs reference PSNR: {psnr:.2f} dB  (expect ~99 if slice correct)")
    Image.fromarray(rgb.astype(np.uint8), 'RGB').save(os.path.join(ROOT, f'rtl_{tag}_rgb.png'))
    print("frac any-zero:", np.mean(np.any(rgb == 0, axis=2)))
else:
    print("rgb len", len(r_rgb))
