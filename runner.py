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
    def __init__(self):
        self.generator = RAGGenerator()
        self.image_dir = "pic_data"
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

    def display_conversation_history(self):
        """Display past conversations in sidebar"""
        with st.sidebar:
            st.markdown("### Conversation History")
            if st.button("Clear History"):
                st.session_state.conversation_history = []
                self.generator.reset_conversation()
                st.experimental_rerun()
            
            # Display previous conversations
            for i, exchange in enumerate(st.session_state.conversation_history[:-1], 1):
                st.markdown(f"**Q{i}: {exchange['query'][:30]}...**")
                with st.expander("View Details"):
                    st.markdown("**Question:**")
                    st.write(exchange['query'])
                    st.markdown("**Answer:**")
                    st.write(exchange['answer'])

    def format_sources_for_display(self, sources: List[Dict]) -> None:
        """Format and display sources in Streamlit"""
        if not sources:
            st.warning("No sources used.")
            return
            
        st.markdown("#### Sources:")
        cols = st.columns(2)
        
        for i, source in enumerate(sources, 1):
            with cols[i % 2]:
                with st.expander(f"Source {i}"):
                    if source['type'] == 'text':
                        st.markdown(f"**Text from page {source['page']}**")
                        st.markdown(f"Preview: {source['content_preview']}")
                    else:
                        st.markdown(f"**Image from page {source['page']}**")
                        image_path = os.path.join(self.image_dir, source['path'])
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Source Image {i}")
                        else:
                            st.warning(f"Image not found: {image_path}")

    def run_interface(self):
        """Run the Streamlit interface"""
        st.title("Multimodal RAG System - Corporate Document Analysis")
        
        # Help section in sidebar
        with st.sidebar:
            st.markdown("### Help")
            st.markdown("""
            **Available Features:**
            - Enter your question in the text input
            - View answer and sources below
            - Sources include both text and images
            - Past conversations in sidebar
            """)
            st.markdown("---")
        
        # Display conversation history in sidebar
        self.display_conversation_history()
        
        st.markdown("---")
        
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
                    
                    # Store in session state
                    st.session_state.conversation_history.append({
                        'query': query,
                        'answer': result['answer'],
                        'sources': result['sources']
                    })
                    # Add console logging for conversation history
                    logger.info("Current Conversation History:")
                    for idx, conv in enumerate(st.session_state.conversation_history, 1):
                        logger.info(f"\nConversation {idx}:")
                        logger.info(f"Q: {conv['query']}")
                        logger.info(f"A: {conv['answer'][:100]}...")  # Show first 100 chars of answer
                    
                    # Display current results
                    st.markdown("### Current Answer")
                    st.write(result['answer'])
                    st.markdown("---")
                    
                    # Display sources for current query
                    self.format_sources_for_display(result['sources'])

            except Exception as e:
                logger.error(f"Error: {e}")
                st.error(f"An error occurred: {str(e)}")

def main():
    interface = StreamlitRAGInterface()
    interface.run_interface()

if __name__ == "__main__":
    main()