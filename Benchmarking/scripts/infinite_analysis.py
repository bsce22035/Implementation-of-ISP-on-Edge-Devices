import os
import time
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from infinite_isp import InfiniteISP
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# --- Configuration ---
RAW_DATA_DIR = "./test/mediatek_raw"
GROUND_TRUTH_DIR = "./test/fujifilm"
CONFIG_PATH = "./config/rgb_config.yml"
RESULTS_DIR = "./performance_results"
SUBSET_SIZE = None  # For testing. Set to None for all.

os.makedirs(RESULTS_DIR, exist_ok=True)

class PerformanceAnalysis:
    def __init__(self, raw_dir, gt_dir, config_path):
        self.raw_dir = Path(raw_dir)
        self.gt_dir = Path(gt_dir)
        self.config_path = config_path
        self.isp = InfiniteISP(str(self.raw_dir), config_path)
        
        # Override config for test data
        self.isp.sensor_info['width'] = 256
        self.isp.sensor_info['height'] = 256
        self.isp.sensor_info['bit_depth'] = 12 # Assume 12- bit
        self.isp.sensor_info['bayer_pattern'] = 'rggb'
        
        self.isp.parm_blc['is_enable'] = True
        self.isp.parm_blc['r_offset'] = 256
        self.isp.parm_blc['gr_offset'] = 256
        self.isp.parm_blc['gb_offset'] = 256
        self.isp.parm_blc['b_offset'] = 256

        self.isp.parm_gmc['is_enable'] = True
        
        # Disable incompatible hardcoded color modules
        self.isp.parm_ccm['is_enable'] = False
        self.isp.parm_awb['is_enable'] = True
        self.isp.parm_wbc['is_enable'] = True
        
    def load_png_raw(self, filename):
        """Loads a PNG file as raw Bayer data, maintaining its true 16-bit values."""
        import cv2
        raw_data = cv2.imread(str(self.raw_dir / filename), cv2.IMREAD_UNCHANGED)
        # Mediatek RAW PNGs in this dataset are already in the 0-4095 range.
        # No bit shifting is required.
        return raw_data.astype(np.uint16)

    def calculate_psnr(self, img1, img2):
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def run(self, limit=None):
        raw_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.png')])
        if limit:
            raw_files = raw_files[:limit]
            
        results = []
        ssim_values = []
        
        print(f"Starting performance analysis on {len(raw_files)} images...")
        
        total_time_taken = 0.0
        
        for filename in tqdm(raw_files):
            try:
                raw_data = self.load_png_raw(filename)
                self.isp.raw = raw_data
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
            
            # Inject raw data into ISP instance (bypassing load_raw)
            self.isp.in_file = Path(filename).stem
            self.isp.out_file = "Out_" + self.isp.in_file
            self.isp.platform["in_file"] = self.isp.in_file
            self.isp.platform["out_file"] = self.isp.out_file
            
            processed_img = []
            
            import util.utils as util_orig
            def mock_save(img_name, output_img, config):
                processed_img.append(output_img)
            
            orig_save = util_orig.save_pipeline_output
            util_orig.save_pipeline_output = mock_save
            
            try:
                # Run once to calculate 3A statistics (Auto White Balance / Auto Exposure)
                self.isp.run_pipeline(visualize_output=False)
                self.isp.load_3a_statistics()
            except Exception as e:
                print(f"Error computing 3A for {filename}: {e}")
                continue
            
            # Measure strictly the ISP execution time for the final pass
            start_time = time.perf_counter()
            try:
                # visualize_output=True triggers the saving mechanism which we mocked to capture the image in memory
                self.isp.run_pipeline(visualize_output=True)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
            finally:
                util_orig.save_pipeline_output = orig_save
            
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            total_time_taken += time_taken
                
            if not processed_img:
                print(f"Failed to process {filename}")
                continue
                
            isp_out = processed_img[0]
            
            # Load ground truth
            try:
                gt_img = plt.imread(self.gt_dir / filename)
                if gt_img.dtype == np.float32:
                    gt_img = (gt_img * 255).astype(np.uint8)
                
                # Calculate metrics
                # Calculate PSNR
                psnr = self.calculate_psnr(gt_img, isp_out)

                # Calculate SSIM
                ssim_val = ssim(
                    gt_img,
                    isp_out,
                    channel_axis=2,
                    data_range=255
                )

                ssim_values.append(ssim_val)
                
                print(f"GT out shape: {gt_img.shape}, dtype: {gt_img.dtype}")
                print(f"ISP out shape: {isp_out.shape}, dtype: {isp_out.dtype}")

                import cv2
                # Save the output image
                cv2.imwrite(os.path.join(RESULTS_DIR, f"output_{filename}"), cv2.cvtColor(isp_out, cv2.COLOR_RGB2BGR))

            except Exception as e:
                psnr = 0.0
                ssim_val = 0.0
                print(f"Error with GT for {filename}: {e}")
            
            fps = 1.0 / time_taken if time_taken > 0 else float('inf')
            print(
                f"{filename} | "
                f"Time={time_taken:.4f}s | "
                f"FPS={fps:.2f} | "
                f"PSNR={psnr:.2f} dB | "
                f"SSIM={ssim_val:.4f}"
            )
            
            results.append({
                'filename': filename,
                'time_taken_sec': time_taken,
                'fps': fps,
                'psnr': psnr,
                'ssim': ssim_val
            })
            
        if not results:
            print("No results generated.")
            return

        # Generate Report
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])

        avg_time = np.mean([r['time_taken_sec'] for r in results])

        overall_fps = (
            len(results) / total_time_taken
            if total_time_taken > 0 else 0
        )
        
        report_path = os.path.join(RESULTS_DIR, "performance_report.txt")
        with open(report_path, "w") as f:
            f.write("Infinite-ISP Performance & Accuracy Analysis Report\n")
            f.write("====================================================\n\n")
            f.write(f"Dataset: Mediatek RAW -> Fujifilm Ground Truth\n")
            f.write(f"Number of images: {len(results)}\n")
            f.write(f"Total Processing Time: {total_time_taken:.4f} seconds\n")
            f.write(f"Average Time Per Image: {avg_time:.4f} seconds\n")
            f.write(f"Overall FPS: {overall_fps:.2f} Frames/Second\n")
            f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.6f}\n\n")
            
            f.write("Individual Results:\n")
            f.write("Filename             | Time(s) | FPS | PSNR(dB) | SSIM\n")
            f.write("---------------------|---------|-----|----------|------\n")
            for r in results:
                fname = r['filename'][:20].ljust(20)
                f.write(
                    f"{fname} | "
                    f"{r['time_taken_sec']:.4f} | "
                    f"{r['fps']:.2f} | "
                    f"{r['psnr']:.2f} | "
                    f"{r['ssim']:.4f}\n"
)
                
        print("\n========== RESULTS ==========")
        print("Number of images:", len(results))
        print("Total Processing Time: {:.4f} sec".format(total_time_taken))
        print("Average Inference Time: {:.6f} sec".format(avg_time))
        print("Overall FPS: {:.2f}".format(overall_fps))
        print("Average PSNR: {:.4f} dB".format(avg_psnr))
        print("Average SSIM: {:.6f}".format(avg_ssim))

        print(f"\nAnalysis complete. Results saved to {RESULTS_DIR}/performance_report.txt")

if __name__ == "__main__":
    analyzer = PerformanceAnalysis(RAW_DATA_DIR, GROUND_TRUTH_DIR, CONFIG_PATH)
    analyzer.run(limit=SUBSET_SIZE)
