import os
import logging
from pathlib import Path
from typing import Optional, Union, List
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import pdfplumber
import torch
from PIL import Image
from io import BytesIO

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exceptions
class EmbeddingError(Exception): pass
class FileProcessingError(EmbeddingError): pass

# Text embedding using SentenceTransformer
def get_hf_text_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list:
    try:
        model = SentenceTransformer(model_name)
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Hugging Face text embedding failed: {e}")
        raise EmbeddingError(f"Failed to generate HF text embedding: {e}")

# Image embedding using CLIP
def get_clip_image_embedding(pil_image: Image.Image) -> list:
    try:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        return outputs.squeeze().tolist()
    except Exception as e:
        logger.error(f"CLIP image embedding failed: {e}")
        raise EmbeddingError(f"Failed to generate image embedding: {e}")

# Extract text from all PDF pages
def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return full_text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise FileProcessingError(f"PDF processing error: {e}")

# Extract tables as flattened text
def extract_tables_from_pdf(pdf_path: Union[str, Path]) -> List[str]:
    tables_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join([", ".join(row) for row in table if row])
                    tables_text.append(table_text)
    except Exception as e:
        logger.warning(f"Failed to extract tables: {e}")
    return tables_text

# Extract images as PIL images
def extract_images_from_pdf(pdf_path: Union[str, Path]) -> List[Image.Image]:
    images = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for image_dict in page.images:
                    x0, top, x1, bottom = image_dict["x0"], image_dict["top"], image_dict["x1"], image_dict["bottom"]
                    bbox = (x0, top, x1, bottom)
                    cropped = page.crop(bbox)
                    img = cropped.to_image(resolution=300).original
                    pil_img = Image.open(BytesIO(img.tobytes()))
                    images.append(pil_img)
    except Exception as e:
        logger.warning(f"Image extraction failed: {e}")
    return images

# Full processor
def process_pdf_file(file_path: Union[str, Path]) -> dict:
    file_path = Path(file_path)
    if not file_path.exists() or file_path.suffix.lower() != '.pdf':
        raise FileProcessingError(f"Invalid file: {file_path}")

    result = {}

    # Text Embedding
    text = extract_text_from_pdf(file_path)
    if text:
        result["text_embedding"] = get_hf_text_embedding(text)

    # Table Embeddings
    table_texts = extract_tables_from_pdf(file_path)
    result["table_embeddings"] = [get_hf_text_embedding(t) for t in table_texts if t]

    # Image Embeddings
    images = extract_images_from_pdf(file_path)
    result["image_embeddings"] = [get_clip_image_embedding(img) for img in images]

    return result

def main():
    pdf_file = "ltimindtree_annual_report.pdf"

    if os.path.exists(pdf_file):
        logger.info(f"Processing PDF: {pdf_file}")
        try:
            embeddings = process_pdf_file(pdf_file)
            logger.info(f"✅ Text embedding vector length: {len(embeddings.get('text_embedding', []))}")
            logger.info(f"📊 Table embeddings: {len(embeddings.get('table_embeddings', []))} tables processed")
            logger.info(f"🖼️ Image embeddings: {len(embeddings.get('image_embeddings', []))} images processed")
        except EmbeddingError as e:
            logger.error(f"Processing failed: {e}")
    else:
        logger.error(f"PDF not found: {pdf_file}")

if __name__ == "__main__":
    main()
