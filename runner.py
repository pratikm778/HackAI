from dotenv import load_dotenv
load_dotenv()  
import os
import logging
from PIL import Image
import openai
from rag_generator import RAGGenerator
from typing import Dict, List, Tuple
import streamlit as st
import pandas as pd
import csv
from io import StringIO
import matplotlib.pyplot as plt
import ast
import re
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
            prompt = f"""
            The following text contains data. Generate Python code using matplotlib or pandas 
            to visualize the data in the most meaningful way. Ensure the code is simple 
            and generates a valid chart or graph:
            
            {answer}
            
            Provide only the Python code as the response.
            """

            # Call GPT to generate the code
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates Python code for data visualization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # Extract the Python code from GPT's response
            raw_code = response.choices[0].message.content.strip()

            # Use regex to extract the Python code block
            match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                st.error("No valid Python code block found in the GPT response.")
                logger.error("No valid Python code block found in the GPT response.")
                return

            # Validate and execute the code
            try:
                ast.parse(code)  # Check if the code is syntactically valid
                exec_globals = {"plt": plt, "pd": pd, "st": st}
                exec(code, exec_globals)
                st.pyplot(plt.gcf())  # Render the current figure
            except Exception as e:
                logger.error(f"Error executing code: {e}")
                st.error(f"Error executing code: {e}")

        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            st.error(f"An error occurred while generating the visualization: {str(e)}")

    def generate_csv_table(self, answer: str):
        """Generate a CSV table from the processed answer"""
        try:
            # Step 1: Get structured data from GPT
            prompt = f"""
            Analyze the following text and create a structured table format:
            1. First, identify appropriate column headers
            2. Then, extract and organize the relevant data into rows
            3. Format your response as a Python dictionary with two keys:
               - 'headers': list of column names
               - 'data': list of lists containing the row values
            
            Text to analyze:
            {answer}

            Ensure the response is structured as valid Python code that creates a dictionary.
            """

            # Get GPT response
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data structuring assistant that converts text into organized tabular data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            # Process GPT response
            raw_code = response.choices[0].message.content.strip()
            match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
            code = match.group(1).strip() if match else raw_code

            # Execute code to get dictionary
            local_dict = {}
            exec(code, {}, local_dict)
            
            # Find the table data dictionary
            table_data = next(
                (v for v in local_dict.values() 
                 if isinstance(v, dict) and 'headers' in v and 'data' in v),
                None
            )

            if not table_data:
                st.error("Could not extract table structure from the response")
                return

            # Step 2: Create and format table
            # Create CSV in memory
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(table_data['headers'])
            writer.writerows(table_data['data'])

            # Create DataFrame
            df = pd.read_csv(StringIO(output.getvalue()))
            
            # Style the DataFrame
            styled_df = df.style.set_properties(**{
                'background-color': '#f0f2f6',
                'color': 'black',
                'border-color': 'white',
                'border-style': 'solid',
                'border-width': '1px',
                'text-align': 'left',
                'font-size': '1rem',
                'padding': '0.5rem'
            })

            # Step 3: Display the table
            st.markdown("### Generated Table")
            st.markdown("""
            <style>
            .stDataFrame {
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stDataFrame td {
                font-family: 'Arial', sans-serif;
                padding: 0.5rem;
            }
            .stDataFrame th {
                background-color: #4a90e2;
                color: white;
                font-weight: bold;
                text-align: center;
                padding: 0.75rem;
            }
            </style>
            """, unsafe_allow_html=True)

            st.dataframe(
                styled_df,
                use_container_width=True,
                height=min(35 * (len(df) + 1), 400),
                hide_index=True
            )

            # Add download button
            st.download_button(
                label="📥 Download CSV",
                data=output.getvalue(),
                file_name="generated_table.csv",
                mime="text/csv",
                key="download_csv",
                help="Click to download the table as a CSV file"
            )

        except Exception as e:
            logger.error(f"Error generating CSV table: {e}")
            st.error(f"An error occurred while generating the table: {str(e)}")

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
        query = st.text_input(
            "Enter your question:", 
            placeholder="e.g., What is the turnover rate for permanent employees?"
        )
        
        if query:
            try:
                with st.spinner("Processing your query..."):
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
                    
                    # Display results and controls
                    st.markdown("### Current Answer")
                    st.write(result['answer'])
                    st.markdown("---")
                    
                    # Add visualization and table generation buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Visualize Data"):
                            self.visualize_with_gpt_code(result['answer'])
                    with col2:
                        if st.button("Generate Table"):
                            self.generate_csv_table(result['answer'])
                    
                    # Display sources
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