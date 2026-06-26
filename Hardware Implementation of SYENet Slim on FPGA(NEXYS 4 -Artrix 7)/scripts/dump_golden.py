"""Dump golden per-op buffer contents in RTL channel-major layout (addr=c*16384+y*128+x)
plus the quantized BUF_IN input, for RTL simulation comparison.
Outputs hex (one int8 byte/line) into sim/ directory."""
import os, glob, numpy as np, torch, torch.nn.functional as F, importlib.util

base = os.path.dirname(__file__)
def L(name, path):
    s = importlib.util.spec_from_file_location(name, os.path.join(base, path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ref = L('ref', 'syenet_reference.py')
qmod = L('qi', 'quantize_isp.py')

ROOT = os.path.join(base, '..')
PKL = os.path.join(ROOT, 'model_best_slim.pkl')
CAL = r"D:/fyp_dataset/test-20251007T082342Z-1-001/test/mediatek_raw"
TEST = r"D:/fyp_dataset/test-20251007T082342Z-1-001/test/mediatek_raw/100.png"
SIM = os.path.join(ROOT, 'sim'); os.makedirs(SIM, exist_ok=True)

m = ref.SYEISPNetS(12); m.load_state_dict(torch.load(PKL, map_location='cpu'), strict=True); m.eval()
W = {k: v.detach().numpy() for k, v in m.state_dict().items()}
A = {}
def rc(n, t): A[n] = max(A.get(n, 0.0), qmod.pmax(t))
with torch.no_grad():
    for fp in sorted(glob.glob(os.path.join(CAL, '*.png')))[:16]:
        x = torch.from_numpy(ref.load_raw(fp)).unsqueeze(0); rc('in', x.numpy())
        c0 = m.head.block1[0](x); rc('hb10', c0.numpy())
        p = m.head.block1[1](c0); rc('hb11', p.numpy())
        h1 = m.head.block1[2](p); rc('h1', h1.numpy())
        h2 = m.head.block2(x); rc('h2', h2.numpy())
        head = h1 * h2 + m.head.bias; rc('head', head.numpy())
        b1 = m.body.block1(head); rc('b1', b1.numpy())
        b2 = m.body.block2(head); rc('b2', b2.numpy())
        body = b1 * b2 + m.body.bias; rc('body', body.numpy())
        g0 = F.adaptive_avg_pool2d(body, 1); rc('gap', g0.numpy())
        a1 = m.att[1](g0); rc('ag1', a1.numpy())
        a2 = m.att[2](a1); rc('ag2', a2.numpy())
        a3 = m.att[3](a2); rc('ag3', a3.numpy())
        att = body * torch.sigmoid(a3); rc('att', att.numpy())
        ps = F.pixel_shuffle(att, 2); rc('ps', ps.numpy())
        rc('rgb', m.tail[1](ps).numpy())
S = {k: (v / 127.0 if v > 0 else 1.0) for k, v in A.items()}

conv = {'hb10': ('head.block1.0.weight', 2, 'in', 'hb10'), 'h1': ('head.block1.2.weight', 1, 'hb11', 'h1'),
        'h2': ('head.block2.weight', 2, 'in', 'h2'), 'b1': ('body.block1.weight', 1, 'head', 'b1'),
        'b2': ('body.block2.weight', 0, 'head', 'b2'), 'ag1': ('att.1.weight', 0, 'gap', 'ag1'),
        'ag3': ('att.3.weight', 0, 'ag2', 'ag3'), 'tail': ('tail.1.weight', 1, 'ps', 'rgb')}
bk = {'hb10': 'head.block1.0.bias', 'h1': 'head.block1.2.bias', 'h2': 'head.block2.bias',
      'b1': 'body.block1.bias', 'b2': 'body.block2.bias', 'ag1': 'att.1.bias', 'ag3': 'att.3.bias', 'tail': 'tail.1.bias'}
order = ['hb10', 'h1', 'h2', 'b1', 'b2', 'ag1', 'ag3', 'tail']
wq = {}; sw = {}; req = {}; biasq = {}
for n in order:
    wk, pad, sin, sout = conv[n]; w = W[wk]; oc = w.shape[0]
    swv = np.abs(w).reshape(oc, -1).max(axis=1) / 127.0; swv[swv == 0] = 1.0; sw[n] = swv
    wq[n] = np.clip(np.round(w / swv[:, None, None, None]), -127, 127).astype(np.int32)
    req[n] = qmod.make_requant_pc(swv * S[sin] / S[sout]); biasq[n] = np.round(W[bk[n]] / (swv * S[sin])).astype(np.int64)
qcu = {}
for nm, (sa, sb, so, k) in {'head': ('h1', 'h2', 'head', 'head.bias'), 'body': ('b1', 'b2', 'body', 'body.bias')}.items():
    ps_ = S[sa] * S[sb]; qcu[nm] = {'M': qmod.make_requant(ps_ / S[so]), 'bias': np.round(W[k].reshape(-1) / ps_).astype(np.int64)}
prelu = {'hb11': np.clip(np.round(W['head.block1.1.weight'] * 128), -128, 127).astype(np.int32),
         'ag2': np.clip(np.round(W['att.2.weight'] * 128), -128, 127).astype(np.int32)}
prelu_M = {'hb11': qmod.make_requant(S['hb10'] / (128.0 * S['hb11'])), 'ag2': qmod.make_requant(S['ag1'] / (128.0 * S['ag2']))}
lut = np.zeros(256, np.int32)
for i in range(256):
    v = (i - 128) * S['ag3']; lut[i] = int(np.clip(round(1 / (1 + np.exp(-v)) * 127), 0, 127))

# ---- forward (correct) capturing each buffer ----
x = qmod.q8(ref.load_raw(TEST), S['in'])
a = qmod.conv_int(x, wq['hb10'], 2) + biasq['hb10'][:, None, None]; hb10 = qmod.requant_pc(a, *req['hb10'])
sl = prelu['hb11'][:, None, None]; v = np.where(hb10 >= 0, hb10 << 7, hb10 * sl).astype(np.int64); hb11 = qmod.requant(v, *prelu_M['hb11'])
a = qmod.conv_int(hb11, wq['h1'], 1) + biasq['h1'][:, None, None]; h1 = qmod.requant_pc(a, *req['h1'])
a = qmod.conv_int(x, wq['h2'], 2) + biasq['h2'][:, None, None]; h2 = qmod.requant_pc(a, *req['h2'])
p = h1.astype(np.int64) * h2.astype(np.int64) + qcu['head']['bias'][:, None, None]; head = qmod.requant(p, *qcu['head']['M'])
a = qmod.conv_int(head, wq['b1'], 1) + biasq['b1'][:, None, None]; b1 = qmod.requant_pc(a, *req['b1'])
a = qmod.conv_int(head, wq['b2'], 0) + biasq['b2'][:, None, None]; b2 = qmod.requant_pc(a, *req['b2'])
p = b1.astype(np.int64) * b2.astype(np.int64) + qcu['body']['bias'][:, None, None]; body = qmod.requant(p, *qcu['body']['M'])
gap_body = body.mean(axis=(1, 2)); gap = np.clip(np.round(gap_body * S['body'] / S['gap']), -127, 127).astype(np.int32)
a = (wq['ag1'].reshape(12, 12) @ gap.reshape(12)).astype(np.int64) + biasq['ag1']; ag1 = qmod.requant_pc(a.reshape(12, 1, 1), *req['ag1']).reshape(12)
sl = prelu['ag2']; v = np.where(ag1 >= 0, ag1 << 7, ag1 * sl).astype(np.int64); ag2 = qmod.requant(v, *prelu_M['ag2'])
a = (wq['ag3'].reshape(12, 12) @ ag2.reshape(12)).astype(np.int64) + biasq['ag3']; ag3 = qmod.requant_pc(a.reshape(12, 1, 1), *req['ag3']).reshape(12)
gv = lut[ag3 + 128]
M_att = qmod.make_requant(S['body'] * (1.0 / 127.0) / S['att'])
att = qmod.requant(body.astype(np.int64) * gv[:, None, None], *M_att)


def whex(name, arr):
    with open(os.path.join(SIM, name), 'w') as f:
        for vv in np.asarray(arr).reshape(-1):
            f.write(f"{int(vv) & 0xFF:02x}\n")

# BUF_IN: [4,128,128] channel-major
whex('buf_in.hex', x.astype(np.int32))
# full 12ch buffers [12,128,128] channel-major
for nm, arr in [('hb10', hb10), ('hb11', hb11), ('h1', h1), ('h2', h2), ('head', head),
                ('b1', b1), ('b2', b2), ('body', body), ('att', att)]:
    whex(f'g_{nm}.hex', arr.astype(np.int32))
# vectors (12 values)
for nm, arr in [('gap', gap), ('ag1', ag1), ('ag2', ag2), ('ag3', ag3), ('gv', gv)]:
    whex(f'g_{nm}.hex', arr.astype(np.int32))
print("wrote golden dumps to", SIM)
print("gv (attention gates):", gv.tolist())
print("ag3:", ag3.tolist())
print("gap:", gap.tolist())
