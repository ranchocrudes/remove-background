import numpy as np

from pathlib import Path

from models.sam_model import process_with_sam_predictor, process_with_sam_mask_quality
from models.nano_banana_model import process_image_gemini
from models.rembg_model import process_with_rembg
from models.grabcut import process_with_grabcut

import urllib.request

def ensure_sam_model_exists():
    model_path = Path(__file__).resolve().parent / "models" / "sam_vit_b_01ec64.pth"
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    if not model_path.exists():
        print("📥 Baixando modelo SAM (1GB+) ... isso pode levar alguns minutos.")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        print("✅ Modelo SAM baixado com sucesso!")
    else:
        print("✅ Modelo SAM já encontrado localmente.")


def list_images_sorted(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file()])

def main():
    print("Main.py execution started...")

    ensure_sam_model_exists()

    PROJECT_ROOT = Path(__file__).resolve().parent

    IMAGES_FOLDER = PROJECT_ROOT / "images"
    ALPHAS_FOLDER = PROJECT_ROOT / "images_alpha"

    images = list_images_sorted(IMAGES_FOLDER)
    alphas = list_images_sorted(ALPHAS_FOLDER)

    if len(images) != len(alphas):
        print("❌ ERRO: número de imagens e alphas não bate!")
        print(f"Images: {len(images)}   Alphas: {len(alphas)}")
        return

    print(f"🔎 Encontradas {len(images)} imagens para processar.\n")

    for idx, (img_path, alpha_path) in enumerate(zip(images, alphas)):
        print(f"\n====== PROCESSANDO #{idx+1} ======")
        print(f"Imagem: {img_path.name}")
        print(f"Alpha : {alpha_path.name}")

        
        # # Rembg – U2NET
        # process_with_rembg(
        #     image_name=img_path.name,
        #     image_alpha_path=str(alpha_path),
        #     model_type="u2net"
        # )

        # # Rembg – ISNet
        # process_with_rembg(
        #     image_name=img_path.name,
        #     image_alpha_path=str(alpha_path),
        #     model_type="isnet-general-use"
        # )

        # # SAM Predictor
        # process_with_sam_predictor(
        #     image_name=img_path.name,
        #     image_alpha_path=str(alpha_path),
        # )
        #         # SAM Mask Quality
        # process_with_sam_mask_quality(
        #     image_name=img_path.name,
        #     image_alpha_path=str(alpha_path),
        # )


        # # GrabCut
        # process_with_grabcut(
        #     image_name=img_path.name,
        #     image_alpha_path=str(alpha_path),
        # )

        # Nano Banana
        # process_image_gemini(
        #     image_name=img_path.name,
        #     image_alpha_path=str(alpha_path)
       
        # )

if __name__ == "__main__":
    main()
