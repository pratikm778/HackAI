import os
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv
from openai import OpenAI
from multimodal_retriever import MultimodalRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    Connects the multimodal retrieval system with OpenAI LLM for generating responses.
    """
    def __init__(self):
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        
        # Initialize retriever
        self.retriever = MultimodalRetriever()
        
        # Default model
        self.model = "gpt-4o-mini"
        
        # Initialize conversation history
        self.conversation_history = []
        self.max_history_length = 10  # Store up to 10 exchanges
    
    def _format_context(self, retrieval_results: Dict) -> str:
        """
        Format retrieved context for the LLM prompt
        
        Args:
            retrieval_results: Results from the multimodal retriever
            
        Returns:
            Formatted context as string
        """
        context = "RELEVANT DOCUMENT SECTIONS:\n\n"
        
        # Add text results
        for i, result in enumerate(retrieval_results['text_results'], 1):
            context += f"TEXT SECTION {i} (Page {result['metadata'].get('page_number', 'unknown')}):\n"
            context += f"{result['text'].strip()}\n\n"
        
        # Add image descriptions if available
        if retrieval_results['image_results']:
            context += "RELEVANT IMAGES:\n"
            for i, result in enumerate(retrieval_results['image_results'], 1):
                context += f"IMAGE {i}: From page {result['page_number']}, path: {result['image_path']}\n"
        
        return context
    
    def _build_prompt(self, query: str, context: str) -> tuple:
        """
        Build a prompt for the LLM including query and context
        
        Args:
            query: User's query
            context: Formatted context from retrieval
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = """You are an AI assistant specialized in analyzing and answering questions about corporate documents.
Use ONLY the information provided in the context below to answer the question. 
If the context doesn't contain enough information to provide a complete answer, acknowledge the limitations and answer based on what is available.
When referencing information from the document, mention the page number where it was found.
If there are images mentioned in the context, reference them if they are relevant to the question.
Make your answers concise and to the point."""
        
        prompt = f"{context}\n\nQUESTION: {query}\n\nANSWER:"
        return system_prompt, prompt
    
    def _prepare_messages_with_history(self, system_prompt: str, user_prompt: str) -> List[Dict]:
        """
        Prepare messages for the API call, including conversation history
        
        Args:
            system_prompt: System prompt
            user_prompt: Current user prompt
            
        Returns:
            List of message dictionaries for the API call
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for entry in self.conversation_history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        
        # Add current query
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def generate_answer(self, query: str, n_text_results: int = 5, n_image_results: int = 3, temperature: float = 0.1) -> Dict:
        """
        Generate an answer for a user query using RAG
        
        Args:
            query: User's query
            n_text_results: Number of text results to retrieve
            n_image_results: Number of image results to retrieve
            temperature: LLM temperature parameter
            
        Returns:
            Dictionary containing the query, answer, and sources used
        """
        try:
            # Get retrieval results
            retrieval_results = self.retriever.hybrid_query(
                query=query,
                n_text_results=n_text_results,
                n_image_results=n_image_results
            )
            
            # Format context from retrieval results
            context = self._format_context(retrieval_results)
            
            # Build prompt
            system_prompt, user_prompt = self._build_prompt(query, context)
            
            # Prepare messages with conversation history
            messages = self._prepare_messages_with_history(system_prompt, user_prompt)
            
            # Generate response from OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            # Extract answer from response
            answer = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": answer
            })
            
            # Limit history size
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history.pop(0)
            
            # Format sources
            sources = []
            for result in retrieval_results['text_results']:
                sources.append({
                    'type': 'text',
                    'page': result['metadata'].get('page_number'),
                    'content_preview': result['text'][:100] + "..."
                })
            
            for result in retrieval_results['image_results']:
                sources.append({
                    'type': 'image',
                    'page': result['page_number'],
                    'path': result['image_path']
                })
            
            return {
                'query': query,
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'query': query,
                'answer': f"Sorry, an error occurred while generating the answer: {str(e)}",
                'sources': []
            }
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history has been reset")


# Example usage
if __name__ == "__main__":
    generator = RAGGenerator()
    
    # Example conversation
    print("\n--- Starting conversation ---")
    
    # First query
    result1 = generator.generate_answer(
        "What were the financial highlights from the last fiscal year?",
        n_text_results=4,
        n_image_results=2
    )
    print("\nQUERY 1:")
    print(result1['query'])
    print("\nANSWER 1:")
    print(result1['answer'])
    
    # Second query (follow-up)
    result2 = generator.generate_answer(
        "Can you tell me more about their digital transformation initiatives?",
        n_text_results=4,
        n_image_results=2
    )
    print("\nQUERY 2:")
    print(result2['query'])
    print("\nANSWER 2:")
    print(result2['answer'])
    
    # Third query (follow-up)
    result3 = generator.generate_answer(
        "Who are the key executives mentioned in those initiatives?",
        n_text_results=4,
        n_image_results=2
    )
    print("\nQUERY 3:")
    print(result3['query'])
    print("\nANSWER 3:")
    print(result3['answer'])
    
    # Reset conversation
    generator.reset_conversation()
    print("\n--- Conversation reset ---")