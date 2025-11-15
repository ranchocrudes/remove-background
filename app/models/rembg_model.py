import time
from datetime import datetime
from rembg import remove, new_session
from PIL import Image
from io import BytesIO
from pathlib import Path

from helpers.report import save_metrics_csv
from helpers.metric import convert_transparent_png_to_bw_mask, metric

_sessions_cache = {}

def get_rembg_session(model_type: str):
    if model_type not in _sessions_cache:
        print(f"Inicializando modelo Rembg: {model_type}")
        _sessions_cache[model_type] = new_session(model_type)
    return _sessions_cache[model_type]

def remove_bg(image_bytes: bytes, model_type: str) -> bytes:
    session = get_rembg_session(model_type)
    return remove(image_bytes, session=session)

def process_with_rembg(image_name: str, image_alpha_path: str, model_type: str) -> Image.Image:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    INPUT_IMAGE_PATH = PROJECT_ROOT / "images" / image_name

    OUTPUT_IMAGE_PATH = (
        PROJECT_ROOT / "results" /f"rembg_{model_type}"  / "colored" / f"rembg_result_{image_name}"
    )
    OUTPUT_IMAGE_ALPHA_PATH = (
        PROJECT_ROOT / "results" /f"rembg_{model_type}" / "alpha" / f"rembg_result_alpha_{image_name}"
    )

    print("--- Starting Rembg Process ---")
    start_time = time.time()

    try:
        with open(INPUT_IMAGE_PATH, "rb") as f:
            input_bytes = f.read()

        output_bytes = remove_bg(input_bytes, model_type=model_type)
        output_image_pil = Image.open(BytesIO(output_bytes)).convert("RGBA")

        elapsed_time_sec = time.time() - start_time
        print(f"Rembg process finished in {elapsed_time_sec:.4f} seconds.")

        OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_IMAGE_ALPHA_PATH.parent.mkdir(parents=True, exist_ok=True)

        colored_png_path = OUTPUT_IMAGE_PATH.with_suffix(".png")
        output_image_pil.save(colored_png_path)

        convert_transparent_png_to_bw_mask(colored_png_path, OUTPUT_IMAGE_ALPHA_PATH)

        iou_score = metric(image_alpha_path, OUTPUT_IMAGE_ALPHA_PATH)

        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_type,
            "image_name": image_name,
            "elapsed_time_sec": round(elapsed_time_sec, 4),
            "alpha_accurate": round(iou_score, 4)
        }

        save_metrics_csv(metrics_data)

        return output_image_pil

    except Exception as e:
        print(f"An error occurred during rembg processing: {e}")
        raise e
