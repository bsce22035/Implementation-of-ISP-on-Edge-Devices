import numpy as np
from PIL import Image

bad = np.array(Image.open("golden_b2corrupt.png").convert("RGB")).astype(int)
fpga = np.array(Image.open("fpga_rgb_v3.png").convert("RGB")).astype(int)
ref = np.array(Image.open("int8_integer_out.png").convert("RGB")).astype(int)

print("fraction of exact-zero pixels (any channel==0):")
for name, im in [("ref", ref), ("b2corrupt", bad), ("fpga", fpga)]:
    z = np.mean(np.any(im == 0, axis=2))
    zall = np.mean(np.all(im == 0, axis=2))
    print(f"  {name:10s} any0={z:.3f}  all0={zall:.3f}  mean={im.mean():.1f}")

print("\ntop-left 6x6 R channel:")
print("ref:\n", ref[:6, :6, 0])
print("b2corrupt:\n", bad[:6, :6, 0])
print("fpga:\n", fpga[:6, :6, 0])

# where does fpga have zeros that b2corrupt does not?
fz = np.all(fpga == 0, axis=2)
bz = np.all(bad == 0, axis=2)
print("\nfpga all-zero pixels:", fz.sum(), " b2corrupt all-zero:", bz.sum())
print("fpga-zero AND b2corrupt-nonzero:", (fz & ~bz).sum())

# Is fpga's zero pattern spatially periodic (pixelshuffle phase)? check parity
print("\nfpga all-zero by (y%2,x%2):")
for a in range(2):
    for b in range(2):
        m = fz[a::2, b::2]
        print(f"  y%2={a} x%2={b}: zero-frac={m.mean():.3f}")
