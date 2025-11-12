import time

from datetime import datetime
from rembg import remove
from PIL import Image
from io import BytesIO
from pathlib import Path

from helpers.report import save_metrics_csv
from helpers.metric import convert_transparent_png_to_bw_mask, metric

def process_with_rembg(image_name: str, image_alpha_path: str) -> Image.Image:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    INPUT_IMAGE_PATH = PROJECT_ROOT / "images" / image_name
    OUTPUT_IMAGE_PATH = PROJECT_ROOT / "results" / "rembg" / "colored" / f"rembg_result_{image_name}.png"
    OUTPUT_IMAGE_ALPHA_PATH = PROJECT_ROOT / "results" / "rembg" / "alpha" / f"rembg_result_alpha_{image_name}.png"

    print("--- Starting Rembg Process ---")
    start_time = time.time()

    try:
        with open(INPUT_IMAGE_PATH, "rb") as f:
            input_bytes = f.read()

        output_bytes = remove_bg(input_bytes)

        output_image_pil = Image.open(BytesIO(output_bytes)).convert("RGBA")

        end_time = time.time()

        elapsed_time_sec = end_time - start_time
        print(f"Rembg process finished in {elapsed_time_sec:.4f} seconds.")

        output_image_pil.save(str(OUTPUT_IMAGE_PATH))

        convert_transparent_png_to_bw_mask(OUTPUT_IMAGE_PATH, OUTPUT_IMAGE_ALPHA_PATH)

        iou_score = metric(image_alpha_path, OUTPUT_IMAGE_ALPHA_PATH)

        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "model": "rembg",
            "image_name": INPUT_IMAGE_PATH,
            "elapsed_time_sec": round(elapsed_time_sec, 4),
            "alpha_accurate": round(iou_score, 4)
        }

        save_metrics_csv(metrics_data)

    except Exception as e:
        print(f"An error occurred during rembg processing: {e}")
        raise e

def remove_bg(image_bytes: bytes) -> bytes:
    return remove(image_bytes)
