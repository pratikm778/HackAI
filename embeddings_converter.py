import os
import sys
import logging
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Union, List, Dict
from google.api_core import exceptions as google_exceptions
from PIL import UnidentifiedImageError
import chromadb
from tqdm import tqdm

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
    load_dotenv()  # Load environment variables
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise APIKeyError("API key not found. Please set GOOGLE_API_KEY in your .env file")
    if api_key == 'your_api_key_here':
        raise APIKeyError("Please replace the default API key with your actual Gemini API key")
    
    # Initialize Gemini API
    genai.configure(api_key=api_key)

def get_text_embedding(text_content: str) -> Optional[list]:
    """Convert text content to embeddings using Gemini API."""
    model = 'models/embedding-001'
    try:
        embedding = genai.embed_content(
            model=model,
            content=text_content,
            task_type="retrieval_document"
        )
        # Return the actual values list instead of the embedding object
        return embedding['embedding']
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
    # Handle direct text input
    if isinstance(file_path, str) and not os.path.exists(file_path):
        return get_text_embedding(file_path)

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

def setup_chromadb() -> chromadb.Collection:
    """Set up and return a ChromaDB collection."""
    # Create a persistent client
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Create or get the collection for text embeddings
    collection = client.get_or_create_collection(
        name="text_embeddings",
        metadata={"description": "Text embeddings from PDF pages"}
    )
    
    return collection

def process_data_folder(data_folder: Union[str, Path], collection) -> None:
    """Process all text files in the data folder and store embeddings in ChromaDB."""
    data_folder = Path(data_folder)
    if not data_folder.exists() or not data_folder.is_dir():
        raise FileProcessingError(f"Data folder not found: {data_folder}")
    
    text_files = sorted(data_folder.glob("text_*.txt"))
    if not text_files:
        raise FileProcessingError("No text files found in data folder")
    
    logger.info(f"Found {len(text_files)} text files to process")
    
    for file_path in tqdm(text_files, desc="Processing text files"):
        try:
            # Read the text content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate embedding
            embedding = get_text_embedding(content)
            if embedding:
                # Extract page number from filename (e.g., "text_7.txt" -> "7")
                page_num = file_path.stem.split('_')[1]
                
                # Add to ChromaDB
                collection.add(
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[{"page": page_num, "source": file_path.name}],
                    ids=[f"page_{page_num}"]
                )
                logger.debug(f"Successfully processed and stored {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

def main():
    try:
        # Validate API key first
        validate_api_key()
        
        # Set up ChromaDB
        logger.info("Setting up ChromaDB...")
        collection = setup_chromadb()
        
        # Process data folder
        data_folder = Path("data")
        logger.info("Processing data folder...")
        process_data_folder(data_folder, collection)
        
        logger.info("Processing complete! Embeddings stored in ChromaDB.")

    except APIKeyError as e:
        logger.critical(f"API Key Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()