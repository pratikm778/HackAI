import os
import re
from pathlib import Path
from typing import List, Dict
from mistralai import Mistral
from dotenv import load_dotenv
import base64
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, pdf_path: str):
        load_dotenv()
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        self.client = Mistral(api_key=self.api_key)
        self.pdf_path = pdf_path
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
    def process_pdf(self) -> None:
        """Process PDF through Mistral OCR and save chunks to data directory"""
        logger.info(f"Processing PDF: {self.pdf_path}")
        
        # Upload PDF
        uploaded_pdf = self._upload_pdf()
        signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
        
        # Process with OCR
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            include_image_base64=True
        )
        
        # Process each page and create chunks
        self._process_pages(ocr_response.pages)
        
    def _upload_pdf(self):
        """Upload PDF to Mistral"""
        with open(self.pdf_path, "rb") as f:
            return self.client.files.upload(
                file={
                    "file_name": Path(self.pdf_path).name,
                    "content": f,
                },
                purpose="ocr"
            )
    
    def _process_pages(self, pages) -> None:
        """Process OCR pages and create text chunks"""
        for page_num, page in enumerate(pages, 1):
            logger.info(f"Processing page {page_num}")
            
            # Get the markdown content for the page
            content = page.markdown
            
            # Create chunks from the page content
            chunks = self._create_chunks(content)
            
            # Save each chunk
            for chunk_num, chunk in enumerate(chunks, 1):
                chunk_filename = f"text_{page_num}_{chunk_num}.txt"
                chunk_path = self.data_dir / chunk_filename
                
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                logger.info(f"Saved chunk: {chunk_filename}")
    
    def _create_chunks(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """Split page content into reasonably-sized chunks"""
        chunks = []
        paragraphs = content.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph is larger than max size, split it
            if para_size > max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_size = 0
                
                for sentence in sentences:
                    sent_size = len(sentence)
                    if temp_size + sent_size <= max_chunk_size:
                        temp_chunk.append(sentence)
                        temp_size += sent_size
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_size = sent_size
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                
            # Normal paragraph handling
            elif current_size + para_size <= max_chunk_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
        
        # Add the last chunk if there is one
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Process PDF
    pdf_path = "ltimindtree_annual_report.pdf"
    processor = PDFProcessor(pdf_path)
    processor.process_pdf()

if __name__ == "__main__":
    main()