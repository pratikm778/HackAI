# filepath: c:\Users\ahnaf\Downloads\HackAI-runner\HackAI-runner\streamlit-frontend\app.py
import sys
import os
import streamlit as st

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_generator import RAGGenerator  # Import the RAGGenerator directly

# Set up the Streamlit app
st.set_page_config(page_title="Multimodal RAG System", layout="wide")

# Custom CSS for deep navy blue background
st.markdown(
    """
    <style>
    body {
        background-color: #001f3f;
        color: white;
    }
    .stTextInput > div > input {
        background-color: #001f3f;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.title("Multimodal RAG System")
st.write("Ask questions about your PDF documents and get answers instantly.")

# Initialize the RAGGenerator
generator = RAGGenerator()

# Chat field
query = st.text_input("Enter your question:", "")
if query:
    try:
        st.write("Processing your query...")
        
        # Generate the answer using RAGGenerator
        result = generator.generate_answer(
            query=query,
            n_text_results=5,
            n_image_results=3,
            temperature=0.1
        )
        
        # Display the answer
        st.subheader("Answer:")
        st.write(result['answer'])
        
        # Display the sources
        st.subheader("Sources:")
        sources = result.get('sources', [])
        if sources:
            for i, source in enumerate(sources, 1):
                if source['type'] == 'text':
                    st.write(f"**Source {i}:** Text from page {source['page']}")
                    st.write(f"Preview: {source['content_preview']}")
                else:
                    # Adjust the image path to be relative to the Streamlit app's working directory
                    image_path = os.path.abspath(os.path.join("..", source['path']))
                    st.write(f"**Source {i}:** Image from page {source['page']}")
                    st.write(f"Path: {source['path']}")
        else:
            st.write("No sources used.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")