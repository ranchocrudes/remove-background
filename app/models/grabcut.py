import cv2
import time
import numpy as np
from datetime import datetime
from PIL import Image
from pathlib import Path

from helpers.report import save_metrics_csv
from helpers.metric import convert_transparent_png_to_bw_mask, metric


def process_with_grabcut(image_name: str, image_alpha_path: str) -> Image.Image:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    INPUT_IMAGE_PATH = PROJECT_ROOT / "images" / image_name

    OUTPUT_IMAGE_PATH = (
        PROJECT_ROOT / "results" / "grabcut" / "colored" / f"grabcut_result_{image_name}"
    )
    OUTPUT_IMAGE_ALPHA_PATH = (
        PROJECT_ROOT / "results" / "grabcut" / "alpha" / f"grabcut_result_alpha_{image_name}"
    )

    print("--- Starting GrabCut Process ---")
    start_time = time.time()

    try:
        
        image = cv2.imread(str(INPUT_IMAGE_PATH))
        if image is None:
            raise Exception(f"Falha ao carregar imagem: {INPUT_IMAGE_PATH}")

        mask = np.zeros(image.shape[:2], np.uint8)

        height, width = image.shape[:2]
        margin = int(min(height, width) * 0.05)
        rect = (margin, margin, width - 2 * margin, height - 2 * margin)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype("uint8")
        
        result = image * mask2[:, :, None]

        result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result_rgba[:, :, 3] = mask2 * 255

        elapsed_time_sec = round(time.time() - start_time, 4)
        print(f"GrabCut process finished in {elapsed_time_sec:.4f} seconds.")

     
        OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_IMAGE_ALPHA_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        colored_png_path = OUTPUT_IMAGE_PATH.with_suffix(".png")
        Image.fromarray(result_rgba).save(colored_png_path)

        convert_transparent_png_to_bw_mask(colored_png_path, OUTPUT_IMAGE_ALPHA_PATH)
        
        iou_score = metric(image_alpha_path, OUTPUT_IMAGE_ALPHA_PATH)
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "model": "grabcut",
            "image_name": image_name,
            "elapsed_time_sec": elapsed_time_sec,
            "alpha_accurate": round(iou_score, 4),
        }
        
        save_metrics_csv(metrics_data)
        
        return Image.fromarray(result_rgba)

    except Exception as e:
        print(f"❌ Erro no GrabCut: {e}")
        raise e
