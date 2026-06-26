#!/usr/bin/env python3
"""
host_isp.py  -  Host interface for the SYENet-Slim ISP FPGA (RAW -> RGB)
========================================================================
Sends a packed/quantised RAW frame to the FPGA and receives the RGB image.

The host does the (float) preprocessing the FPGA shouldn't:
  raw uint16 (256x256)  -> /4095  -> bayer2rggb pack [4,128,128]
                        -> quantise to int8 with in_scale (from quant_params.json)
  -> send 65536 int8 bytes

The FPGA runs the INT8 ISP and returns the final uint8 RGB:
  <- 256x256x3 = 196608 bytes

Packet TX (host->FPGA):  AA 55 AA 55 | 65536 payload bytes | CRC32(payload) LE
Packet RX (FPGA->host):  55 AA 55 AA | 196608 RGB bytes     | CRC32(rgb) LE

Usage:
  python scripts/host_isp.py --port COM15 --raw "D:/.../mediatek_raw/100.png" --out fpga_rgb.png
"""
import os, sys, struct, time, binascii, argparse, json
import numpy as np
try:
    import serial
except ImportError:
    print("pip install pyserial"); sys.exit(1)
from PIL import Image
import importlib.util
_spec = importlib.util.spec_from_file_location('ref', os.path.join(os.path.dirname(__file__),'syenet_reference.py'))
ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)

HDR_TX = b"\xAA\x55\xAA\x55"
HDR_RX = b"\x55\xAA\x55\xAA"
IN_BYTES  = 4*128*128       # 65536
OUT_BYTES = 3*256*256       # 196608

def crc32(b): return binascii.crc32(b) & 0xFFFFFFFF

def prep_input(raw_path, in_scale):
    packed = ref.load_raw(raw_path)                       # [4,128,128] float, /4095
    q = np.clip(np.round(packed / in_scale), -127, 127).astype(np.int8)
    # channel-major flatten: c*H*W + y*W + x  (matches FPGA addressing)
    return q.tobytes()                                    # int8 -> bytes (two's complement)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--raw",  required=True, help="RAW Bayer PNG (256x256 uint16)")
    ap.add_argument("--out",  default="fpga_rgb.png")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--params", default="mem_isp/quant_params.json")
    args = ap.parse_args()

    in_scale = json.load(open(args.params))["in_scale"]
    payload  = prep_input(args.raw, in_scale)
    assert len(payload) == IN_BYTES, f"payload {len(payload)} != {IN_BYTES}"
    packet = HDR_TX + payload + struct.pack("<I", crc32(payload))

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(0.1); ser.reset_input_buffer(); ser.reset_output_buffer()

    print(f"Sending {len(packet)} bytes (RAW packed int8) ...")
    t0 = time.time()
    for i in range(0, len(packet), 1024):
        ser.write(packet[i:i+1024])
    print(f"  sent in {time.time()-t0:.1f}s")

    print(f"Waiting for {OUT_BYTES}-byte RGB result (timeout {args.timeout}s) ...")
    rx = bytearray(); t1 = time.time()
    need = 4 + OUT_BYTES + 4
    while time.time()-t1 < args.timeout and len(rx) < need:
        chunk = ser.read(4096)
        if chunk: rx.extend(chunk)
    ser.close()

    if len(rx) < need:
        print(f"ERROR: got {len(rx)} of {need} bytes."); sys.exit(1)
    if bytes(rx[:4]) != HDR_RX:
        print(f"ERROR: bad RX header {bytes(rx[:4]).hex()}"); sys.exit(1)
    rgb = bytes(rx[4:4+OUT_BYTES])
    rx_crc = struct.unpack_from("<I", rx, 4+OUT_BYTES)[0]
    if crc32(rgb) != rx_crc:
        print(f"WARNING: CRC mismatch calc={crc32(rgb):08x} rx={rx_crc:08x} (saving anyway)")

    # FPGA streams HWC raster order (y, x, c) -> reshape directly to 256x256x3
    arr = np.frombuffer(rgb, np.uint8).reshape(256, 256, 3)
    Image.fromarray(arr, "RGB").save(args.out)
    print(f"Saved RGB -> {args.out}")

if __name__ == "__main__":
    main()
