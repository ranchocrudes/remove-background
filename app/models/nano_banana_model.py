import argparse
import mimetypes
import os
import time
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types
from rembg import remove as rembg_remove
from pathlib import Path

from helpers.report import save_metrics_csv
from helpers.metric import convert_transparent_png_to_bw_mask, metric

MODEL_NAME = "gemini-2.5-flash-image-preview"

def edit_image(
        image_name: str,
        image_alpha_path: str,
        prompt: str,
):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    INPUT_IMAGE_PATH = PROJECT_ROOT / "images" / image_name
    OUTPUT_IMAGE_PATH = PROJECT_ROOT / "results" / "nano_banana" / "colored"
    OUTPUT_IMAGE_ALPHA_PATH = PROJECT_ROOT / "results" / "nano_banana" / "alpha" / f"nano_banana_result_alpha_{image_name}.png"

    api_key = "AIzaSyCN9IX-Y7NFasqki9We5YdcPMaC50tTH1Y"
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    start_time = time.time()

    client = genai.Client(api_key=api_key)

    contents = _load_image_parts([INPUT_IMAGE_PATH])
    contents.append(genai.types.Part.from_text(text=prompt))

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )

    print(f"Editing image: {INPUT_IMAGE_PATH} with prompt: {prompt}")

    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    )

    _process_api_stream_response(stream, OUTPUT_IMAGE_PATH, image_name)

    end_time = time.time()

    elapsed_time_sec = end_time - start_time

    print(f"Image embedding set in {elapsed_time_sec:.4f} seconds.")

    FINAL_OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_PATH / f"nano_banana_result_after_{image_name}.png"

    convert_transparent_png_to_bw_mask(FINAL_OUTPUT_IMAGE_PATH, OUTPUT_IMAGE_ALPHA_PATH)

    iou_score = metric(image_alpha_path, OUTPUT_IMAGE_ALPHA_PATH)

    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "nano_banana",
        "image_name": INPUT_IMAGE_PATH,
        "elapsed_time_sec": round(elapsed_time_sec, 4),
        "alpha_accurate": round(iou_score, 4)
    }

    save_metrics_csv(metrics_data)


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


def _process_api_stream_response(stream, output_dir: str, image_name: str):
    file_index = 0
    file_extension = ".png"
    for chunk in stream:
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        for part in chunk.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                api_image_bytes = part.inline_data.data

                file_name = os.path.join(
                    output_dir,
                    f"nano_banana_result_before_{image_name}{file_extension}",
                )

                _save_binary_file(file_name, api_image_bytes)

                print("Image received from API. Applying rembg...")

                try:
                    transparent_bytes = rembg_remove(api_image_bytes)
                except Exception as e:
                    print(f"Error during rembg processing: {e}")
                    print("Saving the original image from API instead.")
                    transparent_bytes = api_image_bytes

                file_name = os.path.join(
                    output_dir,
                    f"nano_banana_result_after_{image_name}{file_extension}",
                )

                _save_binary_file(file_name, transparent_bytes)
                file_index += 1
            elif part.text:
                print(f"[API Text Response]: {part.text}")


def _save_binary_file(file_name: str, data: bytes):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def _get_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {file_path}")
    return mime_type


def generate_image():
    parser = argparse.ArgumentParser(
        description="Edit an image using Google Generative AI."
    )

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to the input image to edit.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The editing instruction (e.g., 'remove the background').",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the edited images.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    edit_image(
        image_path=args.image,
        prompt=args.prompt,
        output_dir=output_dir,
    )