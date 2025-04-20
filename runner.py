import os
import logging
from PIL import Image
from rag_generator import RAGGenerator
from typing import Dict, List, Tuple
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
import openai

openai.api_key = "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"


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

    def visualize_with_gpt_code(self, answer: str):
        """Use GPT to generate Python code for visualization and execute it"""
        try:
            # Define the prompt for GPT
            prompt = f"""
            The following text contains data. Generate Python code using matplotlib or pandas to visualize the data in the most meaningful way. Ensure the code is simple and generates a valid chart or graph:
            
            {answer}
            
            Provide only the Python code as the response.
            """

            # Call GPT to generate the code
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates Python code for data visualization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            # Extract the Python code from GPT's response
            raw_code = response['choices'][0]['message']['content'].strip()

            # Use regex to extract the Python code block
            match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                st.error("No valid Python code block found in the GPT response.")
                logger.error("No valid Python code block found in the GPT response.")
                return

            # Debugging: Log the extracted code
            #logger.info(f"Extracted code:\n{code}")
            #st.markdown("### Extracted Code")
            #st.code(code, language="python")
            # Validate the generated code
            try:
                ast.parse(code)  # Check if the code is syntactically valid
            except SyntaxError as e:
                logger.error(f"Syntax error in generated code: {e}")
                st.error(f"Syntax error in generated code: {e}")
                return          

            # Execute the validated code
            exec_globals = {"plt": plt, "pd": pd, "st": st}
            try:
                exec(code, exec_globals)
                # Ensure the plot is rendered in Streamlit
                st.pyplot(plt.gcf())  # Render the current figure
            except Exception as exec_error:
                logger.error(f"Error while executing the code: {exec_error}")
                st.error(f"Error while executing the code: {exec_error}")

        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            st.error(f"An error occurred while generating the visualization: {str(e)}")

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
            - Generate tables and charts from structured data
            - Past conversations in sidebar
            """)
            st.markdown("---")
        
        # Display conversation history in sidebar
        self.display_conversation_history()
        
        st.markdown("---")
        
        # Main query interface
        query = st.text_input("Enter your question:", 
                            placeholder="e.g., What is the turnover rate for permanent employees?")
        
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
                    
                    # Display current results
                    st.markdown("### Current Answer")
                    st.write(result['answer'])
                    st.markdown("---")
                    
                    # Add a "Visualize Data" button
                    if st.button("Visualize Data"):
                        # Pass the answer to GPT for visualization
                        self.visualize_with_gpt_code(result['answer'])
                    
                    # Display sources for current query
                    self.format_sources_for_display(result['sources'])

            except Exception as e:
                logger.error(f"Error: {e}")
                st.error(f"An error occurred: {str(e)}")

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

def main():
    interface = StreamlitRAGInterface()
    interface.run_interface()

if __name__ == "__main__":
    main()