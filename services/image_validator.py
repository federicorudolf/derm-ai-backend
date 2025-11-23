"""
Image validation service using CLIP for pre-classification filtering.
Validates that uploaded images are actual skin lesions before classification.
"""
import io
import logging
from typing import Dict, Tuple
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from functools import lru_cache

logger = logging.getLogger(__name__)

# Use CPU for CLIP (lightweight model)
DEVICE = torch.device("cpu")

# CLIP model configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # ~150MB model

# Cached model and processor
_clip_model = None
_clip_processor = None


@lru_cache(maxsize=1)
def load_clip_model():
    """Load and cache CLIP model and processor"""
    global _clip_model, _clip_processor

    if _clip_model is None or _clip_processor is None:
        try:
            logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
            _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            _clip_model.to(DEVICE)
            _clip_model.eval()
            logger.info(f"CLIP model loaded successfully on {DEVICE}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")

    return _clip_model, _clip_processor


def get_clip_model():
    """Get cached CLIP model and processor"""
    return load_clip_model()


def validate_skin_lesion(image_bytes: bytes) -> Dict[str, any]:
    """
    Validate if an image contains a skin lesion using CLIP zero-shot classification.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "reason": str (if invalid),
            "detected_content": str (if invalid),
            "confidence": float,
            "is_skin_lesion_probability": float
        }
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Get CLIP model and processor
        model, processor = get_clip_model()

        # Define candidate labels for zero-shot classification
        # We use multiple phrasings to improve accuracy
        skin_lesion_labels = [
            "a photo of a skin lesion",
            "a photo of a mole on skin",
            "a close-up photo of a skin mark",
            "a medical photo of a skin condition",
            "a dermatology image of a skin spot",
            "a photo of a nevus or melanoma"
        ]

        non_lesion_labels = [
            "a photo of a person",
            "a photo of a face",
            "a photo of an animal",
            "a photo of a cat",
            "a photo of a dog",
            "a photo of food",
            "a photo of a landscape",
            "a photo of an object",
            "a photo of text or document",
            "a screenshot",
            "a photo of nature",
            "a photo of a building",
            "a cartoon or drawing",
            "a photo of a vehicle"
        ]

        # Combine all labels
        all_labels = skin_lesion_labels + non_lesion_labels

        # Process inputs
        inputs = processor(
            text=all_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

        # Calculate skin lesion probability (sum of all skin lesion label probabilities)
        skin_lesion_prob = probs[0, :len(skin_lesion_labels)].sum().item()

        # Get the most likely label overall
        max_prob_idx = probs[0].argmax().item()
        most_likely_label = all_labels[max_prob_idx]
        max_confidence = probs[0, max_prob_idx].item()

        # Validation threshold - require at least 40% combined probability for skin lesion
        # This is conservative to avoid false negatives
        VALIDATION_THRESHOLD = 0.40
        is_valid = skin_lesion_prob >= VALIDATION_THRESHOLD

        if is_valid:
            logger.info(
                f"Image validated as skin lesion "
                f"(skin_lesion_prob: {skin_lesion_prob:.2%}, "
                f"top_label: '{most_likely_label}', confidence: {max_confidence:.2%})"
            )
            return {
                "valid": True,
                "confidence": float(skin_lesion_prob),
                "is_skin_lesion_probability": float(skin_lesion_prob)
            }
        else:
            # Determine what the image likely contains
            detected_content = _extract_content_description(most_likely_label)

            logger.warning(
                f"Image rejected - not a skin lesion "
                f"(skin_lesion_prob: {skin_lesion_prob:.2%}, "
                f"detected: '{detected_content}', confidence: {max_confidence:.2%})"
            )

            return {
                "valid": False,
                "reason": "La imagen no parece ser de un lunar o una lesión cutánea. Por favor, sube una foto de un lunar para poder analizarla.",
                "detected_content": detected_content,
                "confidence": float(max_confidence),
                "is_skin_lesion_probability": float(skin_lesion_prob)
            }

    except Exception as e:
        logger.error(f"Error during image validation: {e}")
        # In case of error, we fail open (allow the image through)
        # This prevents the validation from blocking legitimate requests due to technical issues
        logger.warning("Validation failed, allowing image through as fallback")
        return {
            "valid": True,
            "confidence": 0.0,
            "is_skin_lesion_probability": 0.0,
            "validation_error": str(e)
        }


def _extract_content_description(label: str) -> str:
    """Extract a clean content description from CLIP label"""
    # Remove common prefixes
    content = label.replace("a photo of ", "")
    content = content.replace("a close-up photo of ", "")
    content = content.replace("a medical photo of ", "")
    content = content.replace("a dermatology image of ", "")
    content = content.replace("a ", "")
    content = content.replace("an ", "")

    # Capitalize first letter
    if content:
        content = content[0].upper() + content[1:]

    return content


async def validate_skin_lesion_async(image_bytes: bytes) -> Dict[str, any]:
    """
    Async wrapper for skin lesion validation.
    Currently runs synchronously since CLIP is fast on CPU.
    Can be enhanced with asyncio.to_thread() if needed.
    """
    return validate_skin_lesion(image_bytes)


def preload_validator():
    """Preload the CLIP model during app startup"""
    try:
        logger.info("Preloading CLIP validator model...")
        load_clip_model()
        logger.info("CLIP validator model preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload CLIP validator: {e}")
        raise
