import time
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from pathlib import Path
import mimetypes
import numpy as np
import os
from PIL import Image
from dotenv import load_dotenv

from helpers.report import save_metrics_csv
from helpers.metric import  metric

MODEL_NAME = "gemini-2.5-flash-image-preview"

def convert_gemini_png_to_bw_mask(input_path: str, output_path: str):
    try:
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img)

        black_mask = np.all(arr < 30, axis=-1) 

        mask = np.where(black_mask, 0, 255).astype(np.uint8)

        mask_img = Image.fromarray(mask, mode="L")
        mask_img.save(output_path)

        print(f"Mask saved successfully: {output_path}")

    except Exception as e:
        print(f"Error converting {input_path}: {e}")


def process_image_gemini(image_name: str, image_alpha_path: str, prompt: str = None):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if prompt is None:
        prompt = (
            "Remove the background from the image. "
            "Keep the main object visible and isolated. "
            "Make the background fully black."
        )

    INPUT_IMAGE_PATH = PROJECT_ROOT / "images" / image_name
    OUTPUT_IMAGE_PATH = PROJECT_ROOT / "results" / "nano_banana" / "colored" / f"nano_banana_result_{image_name}"
    OUTPUT_IMAGE_ALPHA_PATH = PROJECT_ROOT / "results" / "nano_banana" / "alpha" / f"nano_banana_result_alpha_{image_name}.png"

    OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_ALPHA_PATH.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv() 

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    start_time = time.time()
    client = genai.Client(api_key=api_key)

    contents = _load_image_parts([INPUT_IMAGE_PATH])
    contents.append(types.Part(text="task: image_editing"))
    contents.append(types.Part(text=prompt))


    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )

    print(f"--- Starting Gemini Process ---\nEditing image: {INPUT_IMAGE_PATH} with prompt: {prompt}")
    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    )

    output_image_pil = _process_api_stream_response(stream, OUTPUT_IMAGE_PATH, OUTPUT_IMAGE_ALPHA_PATH)

    elapsed_time_sec = time.time() - start_time
    print(f"Gemini process finished in {elapsed_time_sec:.4f} seconds.")

    if Path(OUTPUT_IMAGE_ALPHA_PATH).exists():
        iou_score = metric(image_alpha_path, OUTPUT_IMAGE_ALPHA_PATH)
    else:
        print(f"Alpha file not found: {OUTPUT_IMAGE_ALPHA_PATH}")
        iou_score = 0

    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "nano_banana",
        "image_name": image_name,
        "elapsed_time_sec": round(elapsed_time_sec, 4),
        "alpha_accurate": round(iou_score, 4)
    }
    save_metrics_csv(metrics_data)

    return output_image_pil


def _load_image_parts(image_paths: list[str]) -> list[types.Part]:
    parts = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_data = f.read()
        mime_type = _get_mime_type(image_path)
        parts.append(
            types.Part(inline_data=types.Blob(data=image_data, mime_type=mime_type))
        )
    return parts


def _process_api_stream_response(stream, output_image_path: Path, output_image_alpha_path: Path) -> Image.Image:
    """
    Salva a imagem recebida da API e gera o alpha mask.
    """
    for chunk in stream:
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue

        for part in chunk.candidates[0].content.parts:

            if part.inline_data and part.inline_data.data:
                api_image_bytes = part.inline_data.data

                colored_png_path = output_image_path.with_suffix(".png")
                _save_binary_file(colored_png_path, api_image_bytes)

                convert_gemini_png_to_bw_mask(
                    colored_png_path,
                    output_image_alpha_path
                )

                return Image.open(BytesIO(api_image_bytes)).convert("RGBA")

            elif part.text:
                print(f"[API Text Response]: {part.text}")


def _save_binary_file(file_path: Path, data: bytes):
    with open(file_path, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_path}")


def _get_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {file_path}")
    return mime_type


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edit an image using Gemini Generative AI.")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, help="Prompt for the Gemini model.")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    alpha_path = PROJECT_ROOT / "results" / "nano_banana" / "alpha" / f"nano_banana_result_alpha_{args.image}.png"

    process_image_gemini(args.image, alpha_path)
