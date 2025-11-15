import numpy as np
import cv2

from PIL import Image

def convert_transparent_png_to_bw_mask(input_path: str, output_path: str):
    try:
        img = Image.open(input_path).convert("RGBA")
        alpha = np.array(img)[:, :, 3]  # canal alpha

        mask = (alpha > 1).astype(np.uint8) * 255  # binariza corretamente

        mask_img = Image.fromarray(mask, mode="L")
        mask_img.save(output_path)

        print(f"Mask saved successfully: {output_path}")

    except Exception as e:
        print(f"Error to convert {input_path}: {e}")


def calculate_iou(mask_prediction: np.ndarray, mask_true: np.ndarray) -> float:
    mask_prediction = mask_prediction.astype(bool)
    mask_true = mask_true.astype(bool)

    intersection = np.logical_and(mask_prediction, mask_true)

    union = np.logical_or(mask_prediction, mask_true)

    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    if union_area == 0:
        return 1.0

    iou = intersection_area / union_area

    return iou

def metric(template_path: str, predict_path: str):
    try:
        print(template_path, predict_path)
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        predict_img = cv2.imread(predict_path, cv2.IMREAD_GRAYSCALE)

        if template_img is None or predict_img is None:
            raise FileNotFoundError("No image files founded!")

        if template_img.shape != predict_img.shape:
            print(f"Warning: Resizing {template_img.shape}")
            predict_img = cv2.resize(
                predict_img,
                (template_img.shape[1], template_img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        template_result = (template_img > 128)
        predict_result = (predict_img > 128)

        return calculate_iou(predict_result, template_result)
    except Exception as e:
        print(f"Error when loading files: {e}")