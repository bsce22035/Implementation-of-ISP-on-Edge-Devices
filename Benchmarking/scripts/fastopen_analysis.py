import os
import time
import cv2
import numpy as np
from glob import glob
from datetime import datetime

from pipeline import Pipeline
from utils.yacs import Config
from skimage.metrics import structural_similarity as ssim

PNG_DIR = r"./dataset/raw"        # Bayer RAW PNGs
GT_DIR = r"./dataset/fujifilm"    # Ground truth RGB images
CONFIG_PATH = "configs/test.yaml"

OUTPUT_DIR = "./analysis_output"
LOG_FILE = "fast_openisp_png_log.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- PSNR FUNCTION ----------------
def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100

    return 10 * np.log10((255 * 255) / mse)


# ---------------- BENCHMARK ----------------
def benchmark():

    cfg = Config(CONFIG_PATH)
    pipeline = Pipeline(cfg)

    png_files = glob(os.path.join(PNG_DIR, "*.png"))
    png_files.sort()

    times = []
    psnr_list = []
    ssim_list = []

    with open(LOG_FILE, "w") as log:

        log.write("FAST-OPENISP PNG BENCHMARK LOG\n")
        log.write("Date: {}\n".format(datetime.now()))
        log.write("Dataset folder: {}\n".format(PNG_DIR))
        log.write("GT folder: {}\n".format(GT_DIR))
        log.write("Total images: {}\n\n".format(len(png_files)))

        print("Total Images:", len(png_files))

        # ---------------- Warmup ----------------
        if len(png_files) > 0:
            temp = cv2.imread(png_files[0], cv2.IMREAD_UNCHANGED)
            if temp is not None:
                temp = temp.astype(np.uint16)
                pipeline.execute(temp)

        # ---------------- Benchmark Loop ----------------
        for file in png_files:

            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

            if img is None:
                print(f"{file} ERROR reading")
                log.write(f"{file} ERROR reading\n")
                continue

            if len(img.shape) == 3:
                print(f"{file} is RGB, skipping (ISP expects Bayer).")
                continue

            img = img.astype(np.uint16)

            start = time.perf_counter()
            data, _ = pipeline.execute(img)
            end = time.perf_counter()

            infer_time = end - start
            times.append(infer_time)

            name = os.path.basename(file)

            print(f"{name} -> {infer_time:.4f} sec")
            log.write(f"{name} -> {infer_time:.6f} sec\n")

            # ---------------- Save Output ----------------
            output_rgb = data['output']
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

            save_name = name.replace(".png", "_out.png")
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), output_bgr)

            # ---------------- PSNR & SSIM ----------------
            gt_path = os.path.join(GT_DIR, name)

            if os.path.exists(gt_path):

                gt_img = cv2.imread(gt_path)

                if gt_img is not None:

                    # Resize GT if dimensions differ
                    if gt_img.shape != output_bgr.shape:
                        gt_img = cv2.resize(
                            gt_img,
                            (output_bgr.shape[1], output_bgr.shape[0])
                        )

                    # PSNR
                    psnr = calculate_psnr(output_bgr, gt_img)
                    psnr_list.append(psnr)

                    # Convert both images to RGB
                    gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                    out_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

                    # SSIM
                    ssim_val = ssim(
                        gt_rgb,
                        out_rgb,
                        channel_axis=2,
                        data_range=255
                    )

                    ssim_list.append(ssim_val)

                    print(
                        f"   PSNR: {psnr:.4f} dB | SSIM: {ssim_val:.6f}"
                    )

                    log.write(
                        f"   PSNR: {psnr:.4f} dB | SSIM: {ssim_val:.6f}\n"
                    )

                else:
                    print("   GT image read error")
                    log.write("   GT image read error\n")

            else:
                print("   No GT found")
                log.write("   No GT found\n")

        # ---------------- Final Statistics ----------------
        total_images = len(times)

        if total_images > 0:
            total_time = sum(times)
            avg_time = total_time / total_images
            fps = 1.0 / avg_time
            min_time = min(times)
            max_time = max(times)
        else:
            total_time = avg_time = fps = min_time = max_time = 0

        if len(psnr_list) > 0:
            avg_psnr = sum(psnr_list) / len(psnr_list)
        else:
            avg_psnr = 0

        if len(ssim_list) > 0:
            avg_ssim = sum(ssim_list) / len(ssim_list)
        else:
            avg_ssim = 0

        # ---------------- Results ----------------
        print("\nRESULTS")
        print("Total Images Processed:", total_images)
        print("Total Processing Time: {:.6f} sec".format(total_time))
        print("Average Inference Time: {:.6f} sec".format(avg_time))
        print("Min Inference Time: {:.6f} sec".format(min_time))
        print("Max Inference Time: {:.6f} sec".format(max_time))
        print("FPS: {:.2f}".format(fps))
        print("Average PSNR: {:.4f} dB".format(avg_psnr))
        print("Average SSIM: {:.6f}".format(avg_ssim))

        log.write("\nBENCHMARK RESULTS\n")
        log.write("Total Images Processed: {}\n".format(total_images))
        log.write("Total Processing Time: {:.6f} sec\n".format(total_time))
        log.write("Average Inference Time: {:.6f} sec\n".format(avg_time))
        log.write("Min Inference Time: {:.6f} sec\n".format(min_time))
        log.write("Max Inference Time: {:.6f} sec\n".format(max_time))
        log.write("FPS: {:.2f}\n".format(fps))
        log.write("Average PSNR: {:.4f} dB\n".format(avg_psnr))
        log.write("Average SSIM: {:.6f}\n".format(avg_ssim))

    print("\nLog saved to:", LOG_FILE)


if __name__ == "__main__":
    benchmark()