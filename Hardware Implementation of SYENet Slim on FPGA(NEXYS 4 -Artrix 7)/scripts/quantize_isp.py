#!/usr/bin/env python3
"""
quantize_isp.py  -  BIT-EXACT INT8 integer reference + .mem export (HARDWARE SPEC)
==================================================================================
Pure-integer forward pass of SYENet-Slim ISP, identical to what the RTL does:
  INT8 weights/activations (per-tensor symmetric), INT32 MAC, integer requant
  (multiplier M + right shift), QCU (int8*int8 + bias + requant), PReLU (Q0.7),
  sigmoid LUT, PixelShuffle(2), final conv -> uint8 RGB.

The PNG produced here is the reference the FPGA must reproduce.

Exports to mem_isp/:
  w_int8.mem        all conv weights, int8 (one hex byte/line), exec order
  bias_int32.mem    conv biases in accumulator scale (int32, 8 hex/line)
  requant.mem       per conv layer: M(uint16) and shift(uint8)  -> "MMMM SS"
  prelu_int8.mem    PReLU slopes Q0.7 int8  (head.block1.1 then att.2)
  sigmoid_lut.mem   256 entries uint8 (index = att.3 int8 output + 128)
  qcu.mem           QCU requant params for head and body
  scales.txt        all scales + requant constants (human readable, for RTL)
"""
import os, sys, argparse, glob
import numpy as np
try:
    import torch, torch.nn.functional as F
except ImportError:
    print("pip install torch"); sys.exit(1)
import importlib.util
_spec = importlib.util.spec_from_file_location('ref', os.path.join(os.path.dirname(__file__),'syenet_reference.py'))
ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)

# ---------------------------------------------------------------------------
PCT = 99.9   # activation calibration percentile (clip rare outliers)
def amax(x): return float(np.abs(x).max())
def pmax(x): return float(np.percentile(np.abs(x), PCT))
def scale_of(x): m=amax(x); return m/127.0 if m>0 else 1.0
def q8(x, s): return np.clip(np.round(x/s), -127, 127).astype(np.int32)

def make_requant(M_real):
    """Represent real multiplier M_real as M_int * 2^-shift, M_int in 16 bits."""
    if M_real <= 0: return 0, 0
    shift = 0; M = M_real
    while M < (1<<15) and shift < 31:
        M *= 2; shift += 1
    M_int = int(round(M))
    if M_int > 0xFFFF: M_int >>= 1; shift -= 1
    return M_int & 0xFFFF, shift

def make_requant_pc(M_real_vec):
    """Per-channel requant: returns (M_int[C] uint16, shift[C] uint8)."""
    Mi = np.zeros(len(M_real_vec), np.int64); sh = np.zeros(len(M_real_vec), np.int64)
    for i, mr in enumerate(M_real_vec):
        Mi[i], sh[i] = make_requant(float(mr))
    return Mi, sh

def requant_pc(acc, Mi, sh):
    """acc[C,...] * Mi[C] >> sh[C], rounded, clamped int8 (per output channel)."""
    C = acc.shape[0]
    Mi = Mi.reshape([C] + [1]*(acc.ndim-1)); sh = sh.reshape([C] + [1]*(acc.ndim-1))
    prod = acc.astype(np.int64) * Mi
    rnd  = np.where(sh > 0, 1 << np.maximum(sh-1, 0), 0)
    prod = np.where(sh > 0, (prod + rnd) >> np.maximum(sh, 0), prod)
    return np.clip(prod, -127, 127).astype(np.int32)

def requant(acc, M_int, shift):
    """acc(int32 array) * M_int >> shift, rounded, clamped int8."""
    prod = acc.astype(np.int64) * int(M_int)
    if shift > 0:
        prod = (prod + (1 << (shift-1))) >> shift   # round half up
    return np.clip(prod, -127, 127).astype(np.int32)

def conv_int(x_i8, w_i8, pad, stride=1):
    """integer conv via float of small ints (exact for these magnitudes)."""
    xt = torch.from_numpy(x_i8.astype(np.float64)).unsqueeze(0)
    wt = torch.from_numpy(w_i8.astype(np.float64))
    y = F.conv2d(xt, wt, None, stride, pad)
    return y.squeeze(0).numpy()   # float64 holding exact integer sums

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="model_best_slim.pkl")
    ap.add_argument("--out", default="mem_isp")
    ap.add_argument("--calib", default=r"D:/fyp_dataset/test-20251007T082342Z-1-001/test/mediatek_raw")
    ap.add_argument("--ncalib", type=int, default=16)
    ap.add_argument("--test", default=r"D:/fyp_dataset/test-20251007T082342Z-1-001/test/mediatek_raw/100.png")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m = ref.SYEISPNetS(12)
    sd = torch.load(args.pkl, map_location="cpu")
    m.load_state_dict(sd, strict=True); m.eval()
    W = {k: v.detach().numpy() for k,v in m.state_dict().items()}

    # ---------------- calibrate activation scales over images ----------------
    files = sorted(glob.glob(os.path.join(args.calib,"*.png")))[:args.ncalib]
    A = {}
    def rec(n,t): A[n]=max(A.get(n,0.0), pmax(t))   # percentile clips outliers
    with torch.no_grad():
        for fp in files:
            x=torch.from_numpy(ref.load_raw(fp)).unsqueeze(0); rec('in',x.numpy())
            c0=m.head.block1[0](x);                rec('hb10',c0.numpy())
            p =m.head.block1[1](c0);               rec('hb11',p.numpy())
            h1=m.head.block1[2](p);                rec('h1',h1.numpy())
            h2=m.head.block2(x);                   rec('h2',h2.numpy())
            head=h1*h2+m.head.bias;                rec('head',head.numpy())
            b1=m.body.block1(head);                rec('b1',b1.numpy())
            b2=m.body.block2(head);                rec('b2',b2.numpy())
            body=b1*b2+m.body.bias;                rec('body',body.numpy())
            g0=F.adaptive_avg_pool2d(body,1);      rec('gap',g0.numpy())
            a1=m.att[1](g0);                       rec('ag1',a1.numpy())
            a2=m.att[2](a1);                       rec('ag2',a2.numpy())
            a3=m.att[3](a2);                       rec('ag3',a3.numpy())
            g =torch.sigmoid(a3)
            att=body*g;                            rec('att',att.numpy())
            ps=F.pixel_shuffle(att,2);             rec('ps',ps.numpy())
            rgb=m.tail[1](ps);                     rec('rgb',rgb.numpy())
    S={k:(v/127.0 if v>0 else 1.0) for k,v in A.items()}

    # ---------------- quantize weights (per tensor) ----------------
    conv = {  # name -> (weight key, pad, in_scale_name, out_scale_name)
      'hb10':('head.block1.0.weight',2,'in','hb10'),
      'h1'  :('head.block1.2.weight',1,'hb11','h1'),
      'h2'  :('head.block2.weight'  ,2,'in','h2'),
      'b1'  :('body.block1.weight'  ,1,'head','b1'),
      'b2'  :('body.block2.weight'  ,0,'head','b2'),
      'ag1' :('att.1.weight'        ,0,'gap','ag1'),
      'ag3' :('att.3.weight'        ,0,'ag2','ag3'),
      'tail':('tail.1.weight'       ,1,'ps','rgb'),
    }
    bias_key={'hb10':'head.block1.0.bias','h1':'head.block1.2.bias','h2':'head.block2.bias',
              'b1':'body.block1.bias','b2':'body.block2.bias','ag1':'att.1.bias',
              'ag3':'att.3.bias','tail':'tail.1.bias'}
    order=['hb10','h1','h2','b1','b2','ag1','ag3','tail']

    # per-channel (per-output-channel) weight quantisation
    wq={}; sw={}; req={}; biasq={}
    for n in order:
        wk,pad,sin,sout=conv[n]; w=W[wk]                      # [oc,ic,kh,kw]
        oc=w.shape[0]
        swv=np.abs(w).reshape(oc,-1).max(axis=1)/127.0        # per-oc scale
        swv[swv==0]=1.0
        sw[n]=swv
        wq[n]=np.clip(np.round(w/swv[:,None,None,None]),-127,127).astype(np.int32)
        req[n]=make_requant_pc(swv*S[sin]/S[sout])            # (Mi[oc], sh[oc])
        biasq[n]=np.round(W[bias_key[n]]/(swv*S[sin])).astype(np.int64)

    # ---------------- QCU requant params ----------------
    # head = h1*h2 + head.bias ; product scale = S.h1*S.h2 ; out scale = S.head
    qcu={}
    for name,(sa,sb,sout,bk) in {
        'head':('h1','h2','head','head.bias'),
        'body':('b1','b2','body','body.bias')}.items():
        pscale=S[sa]*S[sb]
        qcu[name]={'M':make_requant(pscale/S[sout]),
                   'bias':np.round(W[bk].reshape(-1)/pscale).astype(np.int64)}

    # ---------------- PReLU slopes Q0.7 + per-PReLU requant M ----------------
    # PReLU output scale differs from input (slope squashes negatives), so we
    # rescale: v = x>=0 ? x<<7 : x*slope_q07 ; out = requant(v, M) with
    # M = S_in/(128*S_out).  Unified for both pos/neg paths.
    prelu={'hb11':np.clip(np.round(W['head.block1.1.weight']*128),-128,127).astype(np.int32),
           'ag2' :np.clip(np.round(W['att.2.weight']*128),-128,127).astype(np.int32)}
    prelu_M={'hb11':make_requant(S['hb10']/(128.0*S['hb11'])),
             'ag2' :make_requant(S['ag1'] /(128.0*S['ag2']))}

    # ---------------- sigmoid LUT (index = ag3_int8+128) ----------------
    lut=np.zeros(256,dtype=np.int32)
    for i in range(256):
        v=(i-128)*S['ag3']            # real value of ag3 int8
        lut[i]=int(np.clip(round(1/(1+np.exp(-v))*127),0,127))   # g int8, scale 1/127
    s_g=1.0/127.0

    # =====================================================================
    #              PURE-INTEGER FORWARD  (exactly what RTL does)
    # =====================================================================
    def forward_int(raw_path):
        x=q8(ref.load_raw(raw_path), S['in'])                      # [4,128,128] int8
        # head.block1.0 conv5x5
        a=conv_int(x,wq['hb10'],2)+biasq['hb10'][:,None,None]
        hb10=requant_pc(a,*req['hb10'])
        # PReLU with rescale: v = x>=0 ? x<<7 : x*slope_q07 ; out=requant(v,M)
        sl=prelu['hb11'][:,None,None]
        v=np.where(hb10>=0, hb10<<7, hb10*sl).astype(np.int64)
        hb11=requant(v,*prelu_M['hb11'])
        # head.block1.2 conv3x3
        a=conv_int(hb11,wq['h1'],1)+biasq['h1'][:,None,None]; h1=requant_pc(a,*req['h1'])
        # head.block2 conv5x5
        a=conv_int(x,wq['h2'],2)+biasq['h2'][:,None,None];   h2=requant_pc(a,*req['h2'])
        # QCU head
        p=h1.astype(np.int64)*h2.astype(np.int64)+qcu['head']['bias'][:,None,None]
        head=requant(p,*qcu['head']['M'])
        # body.block1 conv3x3
        a=conv_int(head,wq['b1'],1)+biasq['b1'][:,None,None];  b1=requant_pc(a,*req['b1'])
        # body.block2 conv1x1
        a=conv_int(head,wq['b2'],0)+biasq['b2'][:,None,None];  b2=requant_pc(a,*req['b2'])
        # QCU body
        p=b1.astype(np.int64)*b2.astype(np.int64)+qcu['body']['bias'][:,None,None]
        body=requant(p,*qcu['body']['M'])
        # attention: GAP. mean(body_int8) is in S.body scale; requant to S.gap.
        gap_body=body.mean(axis=(1,2))                            # float, S.body scale
        gap=np.clip(np.round(gap_body*S['body']/S['gap']),-127,127).astype(np.int32)
        a=(wq['ag1'].reshape(12,12)@gap.reshape(12)).astype(np.int64)+biasq['ag1']
        ag1=requant_pc(a.reshape(12,1,1),*req['ag1']).reshape(12)
        sl=prelu['ag2']; v=np.where(ag1>=0,ag1<<7,ag1*sl).astype(np.int64)
        ag2=requant(v,*prelu_M['ag2'])
        a=(wq['ag3'].reshape(12,12)@ag2.reshape(12)).astype(np.int64)+biasq['ag3']
        ag3=requant_pc(a.reshape(12,1,1),*req['ag3']).reshape(12)
        g=lut[ag3+128]                                            # int8 scale 1/127
        # att = body * g  (per channel) ; product scale S.body*s_g ; out S.att
        M_att=make_requant(S['body']*s_g/S['att'])
        att=requant(body.astype(np.int64)*g[:,None,None],*M_att)
        # pixelshuffle(2): 12ch->3ch, 2x   (scale unchanged S.att)
        t=torch.from_numpy(att.astype(np.float64)).unsqueeze(0)
        ps=F.pixel_shuffle(t,2).squeeze(0).numpy().astype(np.int32)  # [3,256,256]
        # tail conv3x3 (3->3)
        a=conv_int(ps,wq['tail'],1)+biasq['tail'][:,None,None]; rgb=requant_pc(a,*req['tail'])
        # to uint8 RGB:  real = rgb*S.rgb, clamp[0,1], *255
        out=np.clip(rgb.astype(np.float64)*S['rgb'],0,1)
        return (out*255+0.5).astype(np.uint8).transpose(1,2,0), M_att

    rgb_u8, M_att = forward_int(args.test)
    from PIL import Image
    Image.fromarray(rgb_u8,'RGB').save('int8_integer_out.png')

    # PSNR vs float reference
    with torch.no_grad():
        yf=m(torch.from_numpy(ref.load_raw(args.test)).unsqueeze(0)).clamp(0,1).squeeze(0).permute(1,2,0).numpy()
    yf8=(yf*255+0.5).astype(np.uint8)
    mse=np.mean((yf8.astype(float)-rgb_u8.astype(float))**2)
    psnr=99 if mse==0 else 10*np.log10(255*255/mse)
    print(f"INTEGER forward vs float: PSNR = {psnr:.1f} dB  -> int8_integer_out.png")

    # ---------------- export .mem + scales ----------------
    def wbytes(path, arr_list):
        with open(path,'w') as f:
            for arr in arr_list:
                for v in np.asarray(arr).reshape(-1):
                    f.write(f"{int(v)&0xFF:02x}\n")
    def w32(path, arr_list):
        with open(path,'w') as f:
            for arr in arr_list:
                for v in np.asarray(arr).reshape(-1):
                    f.write(f"{int(v)&0xFFFFFFFF:08x}\n")
    wbytes(os.path.join(args.out,'w_int8.mem'),   [wq[n] for n in order])
    w32   (os.path.join(args.out,'bias_int32.mem'),[biasq[n] for n in order])
    # requant.mem: per output channel, one "MMMM SS" line, layers in `order`
    with open(os.path.join(args.out,'requant.mem'),'w') as f:
        for n in order:
            Mi,sh=req[n]
            f.write(f"# {n}  ({len(Mi)} output channels)\n")
            for c in range(len(Mi)):
                f.write(f"{int(Mi[c])&0xFFFF:04x} {int(sh[c])&0xFF:02x}\n")
    wbytes(os.path.join(args.out,'prelu_int8.mem'),[prelu['hb11'],prelu['ag2']])
    with open(os.path.join(args.out,'sigmoid_lut.mem'),'w') as f:
        for v in lut: f.write(f"{int(v)&0xFF:02x}\n")
    with open(os.path.join(args.out,'qcu.mem'),'w') as f:
        for name in ['head','body']:
            M,sh=qcu[name]['M']; f.write(f"# {name} M shift then 12 biases(int32)\n{M:04x} {sh:02x}\n")
            for b in qcu[name]['bias']: f.write(f"{int(b)&0xFFFFFFFF:08x}\n")
        M,sh=M_att; f.write(f"# att_scale M shift\n{M:04x} {sh:02x}\n")
    with open(os.path.join(args.out,'scales.txt'),'w') as f:
        f.write("=== activation scales (real = int8 * scale) ===\n")
        for k in sorted(S): f.write(f"  {k:6s} absmax {A[k]:10.4f}  scale {S[k]:.8f}\n")
        f.write("\n=== conv weight scales (per-channel mean) + requant ===\n")
        for n in order:
            Mi,sh=req[n]
            f.write(f"  {n:5s} w_scale[mean {sw[n].mean():.6f} min {sw[n].min():.6f} "
                    f"max {sw[n].max():.6f}]  M[mean {Mi.mean():.0f}] shift[mean {sh.mean():.1f}]\n")
        f.write("\n=== QCU ===\n")
        for name in ['head','body']:
            M,sh=qcu[name]['M']; f.write(f"  {name}: M {M} shift {sh}\n")
        M,sh=M_att; f.write(f"  att_mul: M {M} shift {sh}  (s_g=1/127)\n")
    import json
    with open(os.path.join(args.out,"quant_params.json"),"w") as f:
        json.dump({"in_scale": float(S['in']), "rgb_scale": float(S['rgb']),
                   "img_in": [4,128,128], "img_out": [3,256,256]}, f, indent=2)
    print("Exported .mem + scales + quant_params.json to", os.path.abspath(args.out))

    # ======================================================================
    #   HARDWARE ARTIFACTS for rtl_isp/  (clean $readmemh .mem + isp_params.vh)
    # ======================================================================
    rdir = os.path.join("rtl_isp","mem"); os.makedirs(rdir, exist_ok=True)
    KSZ={'hb10':5,'h1':3,'h2':5,'b1':3,'b2':1,'ag1':1,'ag3':1,'tail':3}
    ICH={'hb10':4,'h1':12,'h2':4,'b1':12,'b2':12,'ag1':12,'ag3':12,'tail':3}
    OCH={'hb10':12,'h1':12,'h2':12,'b1':12,'b2':12,'ag1':12,'ag3':12,'tail':3}

    # weights ROM: concat int8 [oc][ic][ky][kx], record byte base per layer
    wbase={}; wflat=[]; off=0
    for n in order:
        wbase[n]=off; f=wq[n].reshape(-1); wflat.append(f); off+=f.size
    with open(os.path.join(rdir,"weights.mem"),"w") as fh:
        for v in np.concatenate(wflat): fh.write(f"{int(v)&0xFF:02x}\n")
    WLEN=off

    # bias ROM (int32), base per layer in words
    bbase={}; bflat=[]; off=0
    for n in order:
        bbase[n]=off; f=biasq[n].reshape(-1); bflat.append(f); off+=f.size
    with open(os.path.join(rdir,"bias.mem"),"w") as fh:
        for v in np.concatenate(bflat): fh.write(f"{int(v)&0xFFFFFFFF:08x}\n")
    BLEN=off

    # requant ROM: per output channel, word = (M<<8)|shift ; base per layer
    rqbase={}; rqflat=[]; off=0
    for n in order:
        rqbase[n]=off; Mi,sh=req[n]
        for c in range(len(Mi)): rqflat.append(((int(Mi[c])&0xFFFF)<<8)|(int(sh[c])&0xFF))
        off+=len(Mi)
    with open(os.path.join(rdir,"requant.mem"),"w") as fh:
        for v in rqflat: fh.write(f"{v&0xFFFFFF:06x}\n")
    RQLEN=off

    # prelu slopes int8 : hb11(12) then ag2(12)
    with open(os.path.join(rdir,"prelu.mem"),"w") as fh:
        for v in np.concatenate([prelu['hb11'],prelu['ag2']]): fh.write(f"{int(v)&0xFF:02x}\n")
    # sigmoid LUT (256 x uint8)
    with open(os.path.join(rdir,"sigmoid.mem"),"w") as fh:
        for v in lut: fh.write(f"{int(v)&0xFF:02x}\n")
    # qcu bias int32 : head(12) then body(12)
    with open(os.path.join(rdir,"qcubias.mem"),"w") as fh:
        for v in np.concatenate([qcu['head']['bias'],qcu['body']['bias']]):
            fh.write(f"{int(v)&0xFFFFFFFF:08x}\n")

    def vh_layer(n):
        return (f"  // {n}: K{KSZ[n]} {ICH[n]}->{OCH[n]}  wbase {wbase[n]} bbase {bbase[n]} rqbase {rqbase[n]}\n")
    with open(os.path.join("rtl_isp","isp_params.vh"),"w") as f:
        f.write("// AUTO-GENERATED by quantize_isp.py - do not edit\n")
        f.write(f"`define WROM_LEN {WLEN}\n`define BROM_LEN {BLEN}\n`define RQROM_LEN {RQLEN}\n\n")
        for n in order: f.write(vh_layer(n))
        f.write("\n// per-layer base addresses (index order: hb10 h1 h2 b1 b2 ag1 ag3 tail)\n")
        names=order
        def arr(name,vals,w=0):
            f.write(f"`define {name} '{{ "+", ".join(str(int(v)) for v in vals)+" }}\n")
        f.write(f"`define WBASE_HB10 {wbase['hb10']}\n`define WBASE_H1 {wbase['h1']}\n`define WBASE_H2 {wbase['h2']}\n")
        f.write(f"`define WBASE_B1 {wbase['b1']}\n`define WBASE_B2 {wbase['b2']}\n`define WBASE_AG1 {wbase['ag1']}\n`define WBASE_AG3 {wbase['ag3']}\n`define WBASE_TAIL {wbase['tail']}\n")
        f.write(f"`define BBASE_HB10 {bbase['hb10']}\n`define BBASE_H1 {bbase['h1']}\n`define BBASE_H2 {bbase['h2']}\n")
        f.write(f"`define BBASE_B1 {bbase['b1']}\n`define BBASE_B2 {bbase['b2']}\n`define BBASE_AG1 {bbase['ag1']}\n`define BBASE_AG3 {bbase['ag3']}\n`define BBASE_TAIL {bbase['tail']}\n")
        f.write(f"`define RQBASE_HB10 {rqbase['hb10']}\n`define RQBASE_H1 {rqbase['h1']}\n`define RQBASE_H2 {rqbase['h2']}\n")
        f.write(f"`define RQBASE_B1 {rqbase['b1']}\n`define RQBASE_B2 {rqbase['b2']}\n`define RQBASE_AG1 {rqbase['ag1']}\n`define RQBASE_AG3 {rqbase['ag3']}\n`define RQBASE_TAIL {rqbase['tail']}\n")
        # scalar requant params
        hM,hS=qcu['head']['M']; bM,bS=qcu['body']['M']; aM,aS=M_att
        pM0,pS0=prelu_M['hb11']; pM1,pS1=prelu_M['ag2']
        f.write(f"\n`define QCU_HEAD_M {hM}\n`define QCU_HEAD_SH {hS}\n`define QCU_BODY_M {bM}\n`define QCU_BODY_SH {bS}\n")
        f.write(f"`define ATT_MUL_M {aM}\n`define ATT_MUL_SH {aS}\n")
        f.write(f"`define PRELU_HB11_M {pM0}\n`define PRELU_HB11_SH {pS0}\n`define PRELU_AG2_M {pM1}\n`define PRELU_AG2_SH {pS1}\n")
        # GAP requant: body->gap : M for (S.body/S.gap)
        gM,gS=make_requant(S['body']/S['gap'])
        f.write(f"`define GAP_M {gM}\n`define GAP_SH {gS}\n")
        # final RGB output: out_u8 = clamp((rgb_int8 * RGBOUT_M)>>RGBOUT_SH, 0,255)
        oM,oS=make_requant(S['rgb']*255.0)
        f.write(f"`define RGBOUT_M {oM}\n`define RGBOUT_SH {oS}\n")
    print("Exported rtl_isp/mem/*.mem and rtl_isp/isp_params.vh")

if __name__=="__main__":
    main()
