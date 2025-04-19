import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import numpy as np
from PIL import Image
import chromadb
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Import your existing components
from embeddings_processor import GoogleEmbeddingFunction, ContentLabeler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalRetriever:
    """
    A retrieval system that can handle both text and image queries
    and combine results from multiple modalities.
    """
    def __init__(self):
        load_dotenv()
        
        # Configure Google AI
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB client
        self.db_path = "chroma_db"
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize embedding function
        self.embedding_function = GoogleEmbeddingFunction()
        
        # Get references to collections
        self.text_collection = self.client.get_collection(
            name="pdf_embeddings",
            embedding_function=self.embedding_function
        )
        
        # Try to get image collection if it exists
        try:
            self.image_collection = self.client.get_collection(
                name="image_embeddings",
                embedding_function=self.embedding_function
            )
            self.has_image_collection = True
        except:
            logger.warning("Image collection not found. Only text retrieval will be available.")
            self.has_image_collection = False
    
    def query_text(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the text database for similar content"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_function([query])[0]
            
            # Query the collection
            results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = [
                {
                    'text': doc,
                    'metadata': meta,
                    'distance': dist,
                    'modality': 'text'
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying text database: {e}")
            raise
    
    def query_images(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the image database using a text query"""
        if not self.has_image_collection:
            logger.warning("Image collection not available")
            return []
            
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_function([query])[0]
            
            # Query the collection
            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'distances']
            )
            
            # Format results
            formatted_results = [
                {
                    'image_path': meta.get('image_path'),
                    'page_number': meta.get('page_number'),
                    'metadata': meta,
                    'distance': dist,
                    'modality': 'image'
                }
                for meta, dist in zip(
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying image database: {e}")
            return []
    
    def hybrid_query(self, query: str, n_text_results: int = 5, n_image_results: int = 3) -> Dict:
        """
        Perform a hybrid query that returns both text and image results
        
        Args:
            query: The text query
            n_text_results: Number of text results to retrieve
            n_image_results: Number of image results to retrieve
            
        Returns:
            Dictionary containing text and image results
        """
        # Get text results
        text_results = self.query_text(query, n_results=n_text_results)
        
        # Get image results if available
        image_results = self.query_images(query, n_results=n_image_results)
        
        # Combine results
        return {
            'text_results': text_results,
            'image_results': image_results,
            'query': query
        }
    
    def get_related_images_for_text(self, text_result: Dict, n_images: int = 2) -> List[Dict]:
        """Find images that are related to a specific text result"""
        if not self.has_image_collection:
            return []
            
        # Extract page number from text metadata
        page_number = text_result.get('metadata', {}).get('page_number')
        if not page_number:
            return []
            
        # Find images from the same page
        try:
            results = self.image_collection.query(
                query_embeddings=None,
                where={"page_number": page_number},
                n_results=n_images
            )
            
            # Format results
            return [
                {
                    'image_path': meta.get('image_path'),
                    'page_number': meta.get('page_number'),
                    'metadata': meta,
                    'modality': 'image'
                }
                for meta in results['metadatas'][0]
            ]
            
        except Exception as e:
            logger.error(f"Error finding related images: {e}")
            return []


# Example usage
if __name__ == "__main__":
    retriever = MultimodalRetriever()
    
    # Example hybrid query
    results = retriever.hybrid_query(
        "What are the key financial highlights?",
        n_text_results=3,
        n_image_results=2
    )
    
    # Print text results
    print("\nText Results:")
    for i, result in enumerate(results['text_results'], 1):
        print(f"\nResult {i}:")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Page: {result['metadata']['page_number']}")
        print(f"Content Types: {result['metadata'].get('content_types', 'N/A')}")
        print(f"Text Preview: {result['text'][:150]}...")
        
        # Get related images
        related_images = retriever.get_related_images_for_text(result)
        if related_images:
            print(f"Related Images: {', '.join([img['image_path'] for img in related_images])}")
    
    # Print image results
    print("\nImage Results:")
    for i, result in enumerate(results['image_results'], 1):
        print(f"\nImage {i}:")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Page: {result['page_number']}")
        print(f"Image Path: {result['image_path']}")