#!/usr/bin/env python3
"""
batch_infer.py  --  Run batch FPGA inference over a dataset folder
===================================================================
Sends every PNG/JPG image in --dataset to the Nexys-4 FPGA via UART,
collects inference results, and writes a CSV report.

Usage:
    python batch_infer.py --port COM15 --dataset "D:/fyp_dataset/.../test"
                          --class-names "fire,smoke,normal"
                          --out results.csv
                          --limit 10          # optional: test first N images only

Output CSV columns:
    image, source_folder, predicted_class, class_name, confidence,
    logit_0, logit_1, logit_2, status, time_s
"""

import os, sys, struct, binascii, time, argparse, csv
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pip install pyserial")
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: pip install Pillow numpy  -- needed for image loading")
    sys.exit(1)

# ---------------------------------------------------------------------------
HEADER_TX = b"\xAA\x55\xAA\x55"
HEADER_RX = b"\x55\xAA\x55\xAA"
IMG_W = IMG_H = 128
CHANNELS = 4

def crc32(data):
    return binascii.crc32(data) & 0xFFFFFFFF

def prepare_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    rgb = np.array(img, dtype=np.uint8)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    out = np.stack([R, G, G, B], axis=-1)   # RGGB channel-last
    return out.flatten().tobytes()

def build_packet(payload):
    hdr = (HEADER_TX
           + struct.pack("<H", IMG_W)
           + struct.pack("<H", IMG_H)
           + struct.pack("<H", CHANNELS))
    body = hdr + payload
    crc  = struct.pack("<I", crc32(payload))
    return body + crc

def parse_response(data):
    if len(data) < 25:
        raise ValueError(f"short packet ({len(data)} bytes)")
    if data[0:4] != HEADER_RX:
        raise ValueError(f"bad header: {data[0:4].hex()}")
    class_id   = data[4]
    confidence = struct.unpack_from("<i", data, 5)[0]
    logits     = list(struct.unpack_from("<3i", data, 9))
    rx_crc     = struct.unpack_from("<I", data, 21)[0]
    calc_crc   = crc32(data[4:21])
    if calc_crc != rx_crc:
        raise ValueError(f"CRC mismatch calc={calc_crc:08x} rx={rx_crc:08x}")
    return class_id, confidence, logits

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",         required=True,  help="Serial port, e.g. COM15")
    ap.add_argument("--dataset",      required=True,  help="Root folder with images")
    ap.add_argument("--class-names",  default="fire,smoke,normal")
    ap.add_argument("--out",          default="results.csv")
    ap.add_argument("--baud",         type=int, default=115200)
    ap.add_argument("--timeout",      type=int, default=60,  help="Per-image FPGA timeout (s)")
    ap.add_argument("--limit",        type=int, default=0,   help="Max images (0 = all)")
    ap.add_argument("--skip-errors",  action="store_true",   help="Continue on UART/CRC errors")
    args = ap.parse_args()

    class_names = args.class_names.split(",")

    # Collect all images
    dataset = Path(args.dataset)
    exts    = {".png", ".jpg", ".jpeg", ".bmp"}
    images  = sorted([p for p in dataset.rglob("*") if p.suffix.lower() in exts])
    if args.limit > 0:
        images = images[:args.limit]

    total = len(images)
    if total == 0:
        print(f"No images found under {dataset}")
        sys.exit(1)

    bytes_per_pkt = 10 + 65536 + 4
    est_sec_each  = bytes_per_pkt / (args.baud / 10)
    est_total     = total * (est_sec_each + 1)   # +1 for FPGA inference latency
    print(f"Dataset    : {dataset}")
    print(f"Images     : {total}")
    print(f"Port       : {args.port} @ {args.baud}")
    print(f"Output CSV : {args.out}")
    print(f"Est. time  : {est_total/60:.1f} min ({est_total:.0f}s)")
    print()

    # Open serial once for whole batch
    ser = serial.Serial(
        port     = args.port,
        baudrate = args.baud,
        bytesize = serial.EIGHTBITS,
        parity   = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        timeout  = 1.0
    )
    time.sleep(0.1)

    out_path = Path(args.out)
    csvfile  = open(out_path, "w", newline="", encoding="utf-8")
    writer   = csv.writer(csvfile)
    writer.writerow(["image", "source_folder", "predicted_class", "class_name",
                     "confidence", "logit_0", "logit_1", "logit_2", "status", "time_s"])

    ok_count = err_count = 0
    t_batch_start = time.time()

    for idx, img_path in enumerate(images):
        rel   = img_path.relative_to(dataset)
        src   = img_path.parent.name
        fname = img_path.name
        prefix = f"[{idx+1:4d}/{total}] {rel}"

        t0 = time.time()
        try:
            payload = prepare_image(str(img_path))
            packet  = build_packet(payload)

            ser.reset_input_buffer()
            ser.reset_output_buffer()

            # Send
            CHUNK = 1024
            for i in range(0, len(packet), CHUNK):
                ser.write(packet[i:i+CHUNK])

            # Receive
            rx_buf  = bytearray()
            deadline = time.time() + args.timeout
            while time.time() < deadline:
                chunk = ser.read(64)
                if chunk:
                    rx_buf.extend(chunk)
                    if len(rx_buf) >= 25:
                        break

            class_id, conf, logits = parse_response(bytes(rx_buf))
            elapsed  = time.time() - t0
            name     = class_names[class_id] if class_id < len(class_names) else f"cls{class_id}"

            writer.writerow([fname, src, class_id, name, conf,
                             logits[0], logits[1], logits[2], "OK", f"{elapsed:.2f}"])
            csvfile.flush()

            ok_count += 1
            elapsed_batch = time.time() - t_batch_start
            rate = (idx + 1) / elapsed_batch
            eta  = (total - idx - 1) / rate if rate > 0 else 0
            print(f"{prefix}  -> {name} (conf={conf})  {elapsed:.1f}s  ETA {eta/60:.1f}min")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"{prefix}  ERROR: {e}")
            writer.writerow([fname, src, -1, "ERROR", 0, 0, 0, 0, str(e), f"{elapsed:.2f}"])
            csvfile.flush()
            err_count += 1
            if not args.skip_errors:
                print("Stopping. Use --skip-errors to continue past errors.")
                break

    ser.close()
    csvfile.close()

    total_time = time.time() - t_batch_start
    print()
    print("=" * 60)
    print(f"Done: {ok_count} OK, {err_count} errors, {total_time/60:.1f} min total")
    print(f"Results saved to: {out_path.resolve()}")
    print("=" * 60)

    # Quick summary per class
    if ok_count > 0:
        import collections
        counts = collections.Counter()
        with open(out_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "OK":
                    counts[row["class_name"]] += 1
        print("\nPrediction distribution:")
        for cls, cnt in sorted(counts.items()):
            bar = "#" * (cnt * 40 // ok_count)
            print(f"  {cls:10s}: {cnt:5d}  {cnt/ok_count*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
