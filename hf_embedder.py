import os
import logging
from pathlib import Path
from typing import Optional, Union
from sentence_transformers import SentenceTransformer

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

# Hugging Face embedding function
def get_hf_text_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list:
    """Generate text embeddings using Hugging Face sentence transformers."""
    try:
        model = SentenceTransformer(model_name)
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Hugging Face text embedding failed: {e}")
        raise EmbeddingError(f"Failed to generate HF text embedding: {e}")

# File processing
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
            return {"embedding": get_hf_text_embedding(content)}
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            raise FileProcessingError(f"Text file processing error: {e}")
    else:
        raise FileProcessingError(f"Unsupported file type: {file_path.suffix}")

# Entry point
def main():
    text_file = "example.txt"

    if os.path.exists(text_file):
        logger.info("Processing text file with Hugging Face...")
        try:
            embedding = process_file(text_file)
            logger.info(f"Text embedding generated successfully. Vector length: {len(embedding['embedding'])}")
        except EmbeddingError as e:
            logger.error(f"Failed to process text file: {e}")
    else:
        logger.warning(f"'{text_file}' not found. Create it and add some content to test.")

if __name__ == "__main__":
    main()
