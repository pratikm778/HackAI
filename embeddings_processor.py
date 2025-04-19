import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleEmbeddingFunction:
    def __init__(self):
        self.model = "models/embedding-001"
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Google's API
        Args:
            input: List of texts to generate embeddings for
        Returns:
            List of embeddings as float arrays
        """
        embeddings = []
        for text in input:
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise
        return embeddings

class ContentLabeler:
    """Helper class to classify and label content using ML models"""
    
    # Content type categories
    CONTENT_TYPES = {
        'financial': ['revenue', 'profit', 'margin', 'balance', 'income', 'cash flow'],
        'operational': ['operations', 'process', 'workflow', 'efficiency'],
        'strategic': ['strategy', 'vision', 'mission', 'goals'],
        'technological': ['technology', 'digital', 'innovation', 'IT'],
        'sustainability': ['ESG', 'environmental', 'social', 'governance'],
    }
    
    @staticmethod
    def detect_content_type(text: str) -> List[str]:
        """Detect content types based on keyword presence"""
        text_lower = text.lower()
        detected_types = []
        
        for content_type, keywords in ContentLabeler.CONTENT_TYPES.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_types.append(content_type)
                
        return detected_types if detected_types else ['general']
    
    @staticmethod
    def get_metadata(text: str, filename: str) -> Dict:
        """Generate metadata for a chunk of text"""
        # Parse page and chunk numbers from filename
        match = re.match(r'text_(\d+)_(\d+)\.txt', filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        
        page_num, chunk_num = map(int, match.groups())
        
        return {
            'page_number': page_num,
            'chunk_number': chunk_num,
            'content_types': ','.join(ContentLabeler.detect_content_type(text)),
            'timestamp': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'char_count': len(text)
        }

class EmbeddingsProcessor:
    def __init__(self):
        load_dotenv()
        
        # Configure Google AI
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB
        self.db_path = "chroma_db"
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize embedding function using Google's API
        self.embedding_function = GoogleEmbeddingFunction()
        
        collection_name = "pdf_embeddings"
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=collection_name)
            logger.info("Deleted existing collection to ensure correct embedding dimensions")
        except Exception:
            pass  # Collection didn't exist
            
        # Create new collection with Google's embedding function
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "PDF text chunks with embeddings"}
        )
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using Google's API"""
        return self.embedding_function([text])[0]
    
    def process_data_folder(self, data_folder: str = "data") -> None:
        """Process all text files in the data folder"""
        data_path = Path(data_folder)
        if not data_path.exists():
            raise ValueError(f"Data folder not found: {data_folder}")
        
        # Get all text files
        text_files = sorted(data_path.glob("text_*_*.txt"))
        
        for file_path in text_files:
            try:
                # Read the text content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    logger.warning(f"Empty content in {file_path}")
                    continue
                
                # Generate metadata
                metadata = ContentLabeler.get_metadata(content, file_path.name)
                
                # Generate embedding
                embedding = self.get_text_embedding(content)
                
                # Add to ChromaDB
                self.collection.add(
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[file_path.stem]
                )
                
                logger.info(f"Processed {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    def query_similar_content(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the database for similar content"""
        try:
            # Use the same embedding function as the collection
            query_embedding = self.embedding_function([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            return [
                {
                    'text': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            raise

def main():
    try:
        processor = EmbeddingsProcessor()
        processor.process_data_folder()
        
        # Example query
        query_results = processor.query_similar_content(
            "What are the key financial highlights?",
            n_results=3
        )
        
        print("\nExample Query Results:")
        for i, result in enumerate(query_results, 1):
            print(f"\nResult {i}:")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Page: {result['metadata']['page_number']}")
            print(f"Content Types: {result['metadata']['content_types']}")
            print(f"Text Preview: {result['text'][:200]}...")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()