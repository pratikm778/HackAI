import os
from typing import Dict, List
import logging
from dotenv import load_dotenv
from openai import OpenAI
from multimodal_retriever import MultimodalRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LITMindtreeAnalyzer:
    """
    Fetches random facts about LITMindtree and performs financial analysis or storytelling using generative AI.
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
    
    def fetch_random_facts(self, company: str, n_facts: int = 10) -> List[Dict]:
        """
        Fetch random facts about a company using the multimodal retriever.
        
        Args:
            company: Name of the company
            n_facts: Number of random facts to retrieve
            
        Returns:
            List of random facts
        """
        try:
            retrieval_results = self.retriever.hybrid_query(
                query=f"Random facts about {company}",
                n_text_results=n_facts,
                n_image_results=0
            )
            
            facts = []
            for result in retrieval_results['text_results']:
                facts.append({
                    'text': result['text'].strip(),
                    'page': result['metadata'].get('page_number', 'unknown')
                })
            
            return facts
        except Exception as e:
            logger.error(f"Error fetching random facts: {e}")
            return []
    
    def generate_analysis(self, facts: List[Dict], analysis_type: str = "financial analysis") -> Dict:
        """
        Generate financial analysis or storytelling based on random facts.
        
        Args:
            facts: List of random facts
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary containing the analysis and the facts used
        """
        try:
            # Format facts into context
            context = "RANDOM FACTS ABOUT LITMINDTREE:\n\n"
            for i, fact in enumerate(facts, 1):
                context += f"FACT {i} (Page {fact['page']}): {fact['text']}\n\n"
            
            # Build prompt
            system_prompt = f"You are a very creative bestselling short stort teller that writes specialized in {analysis_type}."
            user_prompt = f"{context}\n\nBased on the above facts, provide a detailed {analysis_type}."
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            return {
                'analysis': analysis,
                'facts_used': facts
            }
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            return {
                'analysis': f"Sorry, an error occurred while generating the analysis: {str(e)}",
                'facts_used': facts
            }

# Example usage
if __name__ == "__main__":
    analyzer = LITMindtreeAnalyzer()
    
    # Fetch random facts
    random_facts = analyzer.fetch_random_facts("LITMindtree", n_facts=10)
    
    if not random_facts:
        print("Failed to fetch random facts. Exiting.")
    else:
        # Print random facts
        print("\n--- Random Facts ---")
        for i, fact in enumerate(random_facts, 1):
            print(f"FACT {i}: {fact['text']} (Page {fact['page']})")
        
        # Generate financial analysis
        result = analyzer.generate_analysis(random_facts, analysis_type="financial analysis")
        
        # Print results
        print("\n--- Financial Analysis ---")
        print(result['analysis'])
        
        # Generate storytelling
        storytelling_result = analyzer.generate_analysis(random_facts, analysis_type="storytelling")
        
        # Print storytelling results
        print("\n--- Storytelling ---")
        print(storytelling_result['analysis'])