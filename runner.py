import os
import logging
from PIL import Image
from rag_generator import RAGGenerator
from typing import Dict, List, Tuple
import streamlit as st


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamlitRAGInterface:
    """
    A Streamlit interface for the multimodal RAG system.
    """
    def __init__(self):
        self.generator = RAGGenerator()
        self.image_dir = "pic_data"
        
    def format_sources_for_display(self, sources: List[Dict]) -> None:
        """Format and display sources in Streamlit"""
        if not sources:
            st.warning("No sources used.")
            return
            
        st.subheader("Sources:")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}"):
                if source['type'] == 'text':
                    st.markdown(f"**Text from page {source['page']}**")
                    st.markdown(f"Preview: {source['content_preview']}")
                else:
                    st.markdown(f"**Image from page {source['page']}**")
                    image_path = os.path.abspath(os.path.join("..", source['path']))
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"Source Image {i}")
                    else:
                        st.warning(f"Image not found: {source['path']}")
        
    def run_interface(self):
        """Run the Streamlit interface"""
        st.title("Multimodal RAG System - Corporate Document Analysis")
        st.markdown("---")
        
        # Help section in sidebar
        with st.sidebar:
            st.markdown("### Help")
            st.markdown("""
            **Available Features:**
            - Enter your question in the text input
            - View answer and sources below
            - Sources include both text and images
            - Each source shows the page number and content preview
            """)
        
        # Main query interface
        query = st.text_input("Enter your question:", 
                            placeholder="e.g., What is the relationship between LTIMindtree and IBM?")
        
        if query:
            try:
                with st.spinner("Processing your query..."):
                    # Process query
                    result = self.generator.generate_answer(
                        query=query,
                        n_text_results=5,
                        n_image_results=3,
                        temperature=0.1
                    )
                    
                    # Display results
                    st.subheader("Answer:")
                    st.markdown("---")
                    st.write(result['answer'])
                    
                    # Display sources using the formatting method
                    self.format_sources_for_display(result['sources'])
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                st.error(f"An error occurred: {str(e)}")

def main():
    interface = StreamlitRAGInterface()
    interface.run_interface()

if __name__ == "__main__":
    main()