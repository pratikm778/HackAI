import streamlit as st
from litmindtree_analysis import LITMindtreeAnalyzer
from gtts import gTTS
import os


# Initialize the analyzer
analyzer = LITMindtreeAnalyzer()

# Streamlit App
def main():
    # Inject custom CSS for sidebar background color, text color, and input box styling
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: white;  /* Sidebar background color */
        }
        [data-testid="stSidebar"] label {
            color: black;  /* Black text color for labels */
        }
        [data-testid="stSidebar"] h1 {
            color: black;  /* Black text color for the sidebar title (e.g., "Options") */
        }
        [data-testid="stSidebar"] input {
            background-color: #0D1230;  /* Dark blue background for input box */
            color: white;  /* White text color for input box */
        }
        [data-testid="stSidebar"] select, 
        [data-testid="stSidebar"] textarea {
            background-color: #f0f0f0;  /* Light gray background for dropdowns */
            color: black;  /* Black text color for dropdowns */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("LITMindtree Analysis")
    st.sidebar.image("./LTIMindtree_bg.png", use_container_width=True)  # Add the logo
    st.sidebar.title("Options")
    
    # Sidebar options
    company = st.sidebar.text_input("Enter the company name:", "LITMindtree")
    n_facts = st.sidebar.slider("Number of random facts to fetch:", 1, 20, 10)
    analysis_type = st.sidebar.selectbox("Select analysis type:", ["financial analysis", "storytelling"])
    language = st.sidebar.selectbox("Select language for audio:", ["en", "hi", "es"])  # English, Hindi, Spanish

    # Button to trigger analysis
    if st.sidebar.button("Generate Analysis"):
        # Fetch random facts
        with st.spinner("Fetching random facts..."):
            random_facts = analyzer.fetch_random_facts(company, n_facts=n_facts)
        
        if not random_facts:
            st.error("Failed to fetch random facts. Please try again.")
            return
        
        # Display random facts
        st.subheader("Random Facts")
        for i, fact in enumerate(random_facts, 1):
            st.markdown(f"**FACT {i}** (Page {fact['page']}): {fact['text']}")
        
        # Generate analysis
        with st.spinner(f"Generating {analysis_type}..."):
            result = analyzer.generate_analysis(random_facts, analysis_type=analysis_type)
        
        # Display analysis
        st.subheader(f"{analysis_type.capitalize()}")
        st.write(result['analysis'])

        # Generate Text-to-Speech
        with st.spinner("Generating audio..."):
            tts = gTTS(text=result['analysis'], lang=language, slow=False)
            audio_file = f"{analysis_type}_audio.mp3"
            tts.save(audio_file)
        
        # Play audio
        st.audio(audio_file, format="audio/mp3")
        
        
        # Display facts used
        st.subheader("Facts Used")
        for i, fact in enumerate(result['facts_used'], 1):
            st.markdown(f"**FACT {i}** (Page {fact['page']}): {fact['text']}")

if __name__ == "__main__":
    main()