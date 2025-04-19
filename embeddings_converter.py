import os
import sys
import logging
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Union
from google.api_core import exceptions as google_exceptions
from PIL import UnidentifiedImageError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Base exception for embedding-related errors"""
    pass

class APIKeyError(EmbeddingError):
    """Raised when there are issues with the API key"""
    pass

class FileProcessingError(EmbeddingError):
    """Raised when there are issues processing files"""
    pass

def validate_api_key() -> None:
    """Validate that the API key is present and properly configured"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise APIKeyError("API key not found. Please set GOOGLE_API_KEY in your .env file")
    if api_key == 'your_api_key_here':
        raise APIKeyError("Please replace the default API key with your actual Gemini API key")

def get_text_embedding(text_content: str) -> Optional[dict]:
    """Convert text content to embeddings using Gemini API."""
    model = 'models/embedding-001'
    try:
        embedding = genai.embed_content(
            model=model,
            content=text_content,
            task_type="retrieval_document"
        )
        return embedding
    except google_exceptions.InvalidArgument as e:
        logger.error(f"Invalid argument for text embedding: {e}")
        raise EmbeddingError(f"Invalid input for text embedding: {e}")
    except google_exceptions.PermissionDenied as e:
        logger.error(f"API permission denied: {e}")
        raise APIKeyError(f"API authentication failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in text embedding: {e}")
        raise EmbeddingError(f"Failed to generate text embedding: {e}")

def get_image_embedding(image_path: Union[str, Path]) -> Optional[dict]:
    """Convert image to embeddings using Gemini API."""
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        image = Image.open(image_path)
        response = model.get_response(image)
        return response.candidates[0]
    except UnidentifiedImageError as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise FileProcessingError(f"Invalid or corrupted image file: {e}")
    except google_exceptions.PermissionDenied as e:
        logger.error(f"API permission denied: {e}")
        raise APIKeyError(f"API authentication failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in image embedding: {e}")
        raise EmbeddingError(f"Failed to generate image embedding: {e}")

def process_file(file_path: Union[str, Path]) -> Optional[dict]:
    """Process a file and return its embedding based on file type."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileProcessingError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise FileProcessingError(f"Not a file: {file_path}")
    
    # Handle text files
    if file_path.suffix.lower() in ['.txt', '.md', '.py', '.json']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return get_text_embedding(content)
        except UnicodeDecodeError as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            raise FileProcessingError(f"Error reading text file (encoding issue): {e}")
        except IOError as e:
            logger.error(f"IO error reading file {file_path}: {e}")
            raise FileProcessingError(f"Error reading file: {e}")
    
    # Handle image files
    elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
        return get_image_embedding(file_path)
    
    else:
        msg = f"Unsupported file type: {file_path.suffix}"
        logger.warning(msg)
        raise FileProcessingError(msg)

def main():
    try:
        # Validate API key first
        validate_api_key()
        
        # Example usage
        text_file = "example.txt"
        image_file = "example.jpg"
        
        # Process text file
        if os.path.exists(text_file):
            logger.info("Processing text file...")
            try:
                embedding = process_file(text_file)
                logger.info(f"Text embedding generated successfully. Shape: {len(embedding.values)}")
            except EmbeddingError as e:
                logger.error(f"Failed to process text file: {e}")
        
        # Process image file
        if os.path.exists(image_file):
            logger.info("Processing image file...")
            try:
                embedding = process_file(image_file)
                logger.info("Image embedding generated successfully")
            except EmbeddingError as e:
                logger.error(f"Failed to process image file: {e}")

    except APIKeyError as e:
        logger.critical(f"API Key Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()