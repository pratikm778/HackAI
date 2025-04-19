import os
import fitz  # PyMuPDF
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles all PDF processing operations including text extraction and image extraction
    """
    def __init__(self, output_dir: str = "data", image_dir: str = "pic_data"):
        self.output_dir = Path(output_dir)
        self.image_dir = Path(image_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.image_dir.mkdir(exist_ok=True)

    def process_pdf(self, pdf_path: str, chunk_size: int = 1000) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a PDF file to extract both text and images
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Number of characters per text chunk
            
        Returns:
            Tuple of (text_chunks, image_info)
        """
        logger.info(f"Processing PDF: {pdf_path}")
        text_chunks = []
        image_info = []

        try:
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num, page in enumerate(doc, 1):
                # Extract text
                text = page.get_text()
                if text.strip():
                    chunks = self._split_text(text, chunk_size)
                    for i, chunk in enumerate(chunks, 1):
                        chunk_info = {
                            'text': chunk,
                            'page_number': page_num,
                            'chunk_number': i,
                            'file_path': pdf_path
                        }
                        text_chunks.append(chunk_info)
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list, 1):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Save image
                        image = Image.open(io.BytesIO(image_bytes))
                        image_filename = f"page_{page_num}_img_{img_index}.png"
                        image_path = self.image_dir / image_filename
                        image.save(image_path)
                        
                        # Record image info
                        img_info = {
                            'page_number': page_num,
                            'image_number': img_index,
                            'path': str(image_path),
                            'width': image.width,
                            'height': image.height,
                            'file_path': pdf_path
                        }
                        image_info.append(img_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_index} on page {page_num}: {e}")
            
            return text_chunks, image_info
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately equal size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word)
            if current_size + word_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size + 1  # +1 for space
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def save_text_chunks(self, chunks: List[Dict]) -> None:
        """Save text chunks to individual files"""
        for chunk in chunks:
            filename = f"text_{chunk['page_number']}_{chunk['chunk_number']}.txt"
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(chunk['text'])

def main():
    processor = PDFProcessor()
    
    # Process PDF file
    pdf_path = "ltimindtree_annual_report.pdf"  # Replace with your PDF
    if os.path.exists(pdf_path):
        try:
            text_chunks, image_info = processor.process_pdf(pdf_path)
            processor.save_text_chunks(text_chunks)
            logger.info(f"Processed {len(text_chunks)} text chunks and {len(image_info)} images")
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
    else:
        logger.error(f"PDF file not found: {pdf_path}")

if __name__ == "__main__":
    main()