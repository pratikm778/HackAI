import os
import logging
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom error classes
class EmbeddingError(Exception):
    pass

class FileProcessingError(EmbeddingError):
    pass

# Initialize the BGE-M3 model and processor
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load text components
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    text_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # Load image components (assuming the same model can handle both)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    logger.info(f"Successfully loaded {MODEL_NAME} model on {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise EmbeddingError(f"Model loading failed: {e}")

def get_text_embedding(text: str) -> list:
    """Generate text embeddings using BAAI/bge-m3 model."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = text_model(**inputs)
        # Use the [CLS] token embedding as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        raise EmbeddingError(f"Failed to generate text embedding: {e}")

def get_image_embedding(image_path: Union[str, Path]) -> list:
    """Generate image embeddings using BAAI/bge-m3 model."""
    try:
        image = Image.open(image_path)
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = text_model(**inputs)
        # Use the [CLS] token embedding as the image embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Image embedding failed: {e}")
        raise EmbeddingError(f"Failed to generate image embedding: {e}")

def process_file(file_path: Union[str, Path]) -> Optional[dict]:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileProcessingError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise FileProcessingError(f"Not a file: {file_path}")

    if file_path.suffix.lower() in ['.txt', '.md', '.py', '.json']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"embedding": get_text_embedding(content)}
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            raise FileProcessingError(f"Text file processing error: {e}")
    elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        try:
            return {"embedding": get_image_embedding(file_path)}
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise FileProcessingError(f"Image file processing error: {e}")
    else:
        raise FileProcessingError(f"Unsupported file type: {file_path.suffix}")

# Entry point
def main():
    test_file = "test.txt"  # Change this to test with different files
    
    if os.path.exists(test_file):
        logger.info(f"Processing file: {test_file}")
        try:
            result = process_file(test_file)
            logger.info("Embedding generated successfully")
            logger.info(f"Embedding length: {len(result['embedding'])}")
        except (EmbeddingError, FileProcessingError) as e:
            logger.error(f"Failed to process file: {e}")

if __name__ == "__main__":
    main()