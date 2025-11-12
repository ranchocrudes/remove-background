import torch
import cv2
import numpy as np
import time
import argparse
import mimetypes
import os

from PIL import Image
from pathlib import Path
from google import genai
from google.genai import types

from models.sam_model import process_with_sam_predictor, process_with_sam_mask_quality
from models.nano_banana_model import edit_image
from models.rembg_model import process_with_rembg

from helpers.metric import convert_transparent_png_to_bw_mask

def main():
    print("Main.py execution started...")

    PROJECT_ROOT = Path(__file__).resolve().parent

    IMAGE_NAME = "gato-n-sei.png"
    IMAGE_ALPHA_NAME = "gato-n-sei-alpha.png"

    INPUT_IMAGE_ALPHA_PATH = PROJECT_ROOT / "alpha" / IMAGE_ALPHA_NAME

    try:
        print(f"\nStarting Process...")

        # U²-NET
        process_with_rembg(
            image_name=str(IMAGE_NAME),
            image_alpha_path=str(INPUT_IMAGE_ALPHA_PATH),
        )

        # SAM Predictor
        process_with_sam_predictor(
            image_name=str(IMAGE_NAME),
            image_alpha_path=str(INPUT_IMAGE_ALPHA_PATH),
        )

        # SAM Mask Quality
        process_with_sam_mask_quality(
            image_name=str(IMAGE_NAME),
            image_alpha_path=str(INPUT_IMAGE_ALPHA_PATH),
        )

        # Nano Banana
        edit_image(
            image_name=str(IMAGE_NAME),
            image_alpha_path=str(INPUT_IMAGE_ALPHA_PATH),
            prompt="Remove image background and turn background to black (RGB 000)",
        )

    except Exception as e:
        print(f"\nAn unexpected error occurred in main.py: {e}")

if __name__ == "__main__":
    main()