import sys, os, numpy as np
ROOT = os.path.join(os.path.dirname(__file__), '..'); SIM = os.path.join(ROOT, 'sim')


def rd(p):
    v = [int(x, 16) for x in open(p) if x.strip()]
    a = np.array(v, np.int32); return np.where(a >= 128, a - 256, a)


golden = sys.argv[1]   # e.g. g_b2.hex
rtlfile = sys.argv[2]  # e.g. rtl_b2_m0.mem
g = rd(os.path.join(SIM, golden)); r = rd(os.path.join(SIM, rtlfile))
n = min(len(g), len(r))
eq = np.array_equal(g[:n], r[:n])
print(f"{golden} vs {rtlfile}: match={eq}  lens g={len(g)} r={len(r)}")
if not eq:
    d = np.where(g[:n] != r[:n])[0]
    print(f"  mismatches: {len(d)}/{n} ({100*len(d)/n:.1f}%)")
    for idx in d[:6]:
        c = idx // 16384; p = idx % 16384; y = p // 128; x = p % 128
        print(f"   idx{idx} c={c} y={y} x={x}  golden={g[idx]} rtl={r[idx]}")
    perc = [int(np.sum(g[c*16384:(c+1)*16384] != r[c*16384:(c+1)*16384])) for c in range(min(12, n//16384))]
    print("  per-channel mismatch:", perc)
