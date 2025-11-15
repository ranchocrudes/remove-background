import torch
import cv2
import numpy as np
import time
from datetime import datetime
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path

from helpers.report import save_metrics_csv
from helpers.metric import convert_transparent_png_to_bw_mask, metric

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

def refine_mask(mask_bool: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    print(f"Refining mask with {kernel_size}x{kernel_size} kernel...")

    mask_binary = mask_bool.astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask_open = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

    mask_refined = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    return mask_refined.astype(bool)

def apply_mask_to_image(image_pil, mask):
     image_rgba = image_pil.convert("RGBA")
     img_array = np.array(image_rgba)

     alpha_channel = np.full((img_array.shape[0], img_array.shape[1]), 255, dtype=np.uint8)

     alpha_channel[~mask] = 0

     img_array[..., 3] = alpha_channel

     return Image.fromarray(img_array)

def process_with_sam_predictor(image_name: str, image_alpha_path: str):
    INPUT_IMAGE_PATH = PROJECT_ROOT / "images" / image_name
    OUTPUT_IMAGE_PATH = PROJECT_ROOT / "results" / "sam_predictor" / "colored" / f"sam_predictor_result_{image_name}.png"
    OUTPUT_IMAGE_ALPHA_PATH = PROJECT_ROOT / "results" / "sam_predictor" / "alpha" / f"sam_predictor_result_alpha_{image_name}.png"

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_ALPHA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("--- Starting SAM Predictor Process ---")

    image_bgr = cv2.imread(str(INPUT_IMAGE_PATH))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {INPUT_IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running SAM on device: {device}")

    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
    sam.to(device)
    print("SAM model loaded.")

    predictor = SamPredictor(sam)

    start_time = time.time()
    predictor.set_image(image_rgb)
    elapsed_time_sec = time.time() - start_time

    h, w, _ = image_rgb.shape
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    best_mask = masks[np.argmax(scores)]

    original_image_pil = Image.open(INPUT_IMAGE_PATH)
    result_image = apply_mask_to_image(original_image_pil, best_mask)
    colored_png_path = OUTPUT_IMAGE_PATH.with_suffix(".png")
    result_image.save(str(OUTPUT_IMAGE_PATH))

    convert_transparent_png_to_bw_mask(colored_png_path, OUTPUT_IMAGE_ALPHA_PATH)

    iou_score = metric(image_alpha_path, OUTPUT_IMAGE_ALPHA_PATH)

    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "sam_predictor",
        "image_name": image_name,
        "elapsed_time_sec": round(elapsed_time_sec, 4),
        "alpha_accurate": round(iou_score, 4)
    }

    save_metrics_csv(metrics_data)

    print("--- SAM Predictor Finished ---")


def process_with_sam_mask_quality(image_name: str, image_alpha_path: str,
                                  iou_threshold: float = 0.88,
                                  stability_threshold: float = 0.92):

    print(f"--- Starting SAM Mask Quality Process ---")

    INPUT_IMAGE_PATH = PROJECT_ROOT / "images" / image_name
    OUTPUT_IMAGE_PATH = PROJECT_ROOT / "results" / "sam_mask_quality" / "colored" / f"sam_mask_quality_result_{image_name}.png"
    OUTPUT_IMAGE_ALPHA_PATH = PROJECT_ROOT / "results" / "sam_mask_quality" / "alpha" / f"sam_mask_quality_result_alpha_{image_name}.png"

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_ALPHA_PATH.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(INPUT_IMAGE_PATH))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {INPUT_IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2
    )

    start_time = time.time()
    masks = mask_generator.generate(image_rgb)
    elapsed_time_sec = time.time() - start_time

    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    shape = sorted_masks[0]['segmentation'].shape

    final_foreground_mask = np.zeros(shape, dtype=bool)

    kept = 0
    for m_dict in sorted_masks[1:]:
        if m_dict['predicted_iou'] > iou_threshold and m_dict['stability_score'] > stability_threshold:
            final_foreground_mask |= m_dict['segmentation']
            kept += 1

    original_image_pil = Image.open(INPUT_IMAGE_PATH)
    result_image = apply_mask_to_image(original_image_pil, final_foreground_mask)

    result_image.save(str(OUTPUT_IMAGE_PATH))
    colored_png_path = OUTPUT_IMAGE_PATH.with_suffix(".png")
    convert_transparent_png_to_bw_mask(colored_png_path, OUTPUT_IMAGE_ALPHA_PATH)

    iou_score = metric(image_alpha_path, OUTPUT_IMAGE_ALPHA_PATH)

    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "sam_mask_quality",
        "image_name": image_name,
        "elapsed_time_sec": round(elapsed_time_sec, 4),
        "alpha_accurate": round(iou_score, 4)
    }

    save_metrics_csv(metrics_data)

    print("--- SAM Mask Quality Finished ---")
