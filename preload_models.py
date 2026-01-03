"""
Preload models during Docker build to avoid downloading during startup
"""
import logging
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preload_clip_model():
    """Download CLIP model during build time"""
    try:
        model_name = "openai/clip-vit-base-patch32"
        logger.info(f"Downloading CLIP model: {model_name}")

        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)

        logger.info(f"✓ CLIP model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download CLIP model: {e}")
        return False

if __name__ == "__main__":
    success = preload_clip_model()
    if not success:
        exit(1)
    logger.info("All models preloaded successfully")
