import streamlit as st

# ----------- SESSION SETUP -----------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ----------- MAIN PAGE SWITCHING -----------
def main():
    # Add custom CSS for light blue background and white text
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #001f3f; /* Light blue background */
            color: #ffffff; /* White text */
        }

        /* Styling for feature cards */
        .feature-card {
            background-color: rgba(255, 255, 255, 0.2); /* Semi-transparent white */
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }

        /* Centered header styling */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }

        .header-container img {
            width: 100px;
            height: auto;
        }

        .header-container h1 {
            font-size: 3rem;
            color: #ffffff;
            margin: 0;
        }

        h2, h3 {
            color: #ffffff; /* Ensure headers are white */
        }

        p {
            color: #ffffff; /* Ensure paragraphs are white */
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(50, 50, 50, 0.8); /* Darker sidebar background */
            color: #ffffff; /* White text */
        }

        [data-testid="stSidebar"] h3 {
            color: #ffffff; /* Sidebar title text color */
            text-align: center;
        }

        [data-testid="stSidebar"] p {
            color: #ffffff; /* Sidebar paragraph text color */
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar content
   
    # Header with logo and app name
    st.markdown(
        """
        <div class="header-container">
            <h1>EpicRAG<h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Feature cards
    st.markdown(
        """
        <div class="feature-card">
            <h2>🌐 MultiModal RAG Agent</h2>
            <p>Our pipeline processes PDFs, chunks and vectorizes them, and retrieves relevant chunks for your queries. This enables seamless document understanding and retrieval for your business needs.</p>
        </div>

        <div class="feature-card">
            <h2>🏢 About LTIMindtree</h2>
            <p>LTIMindtree is a global technology consulting and digital solutions company that enables enterprises to reimagine business models, accelerate innovation, and maximize growth. We are proud to integrate cutting-edge AI solutions into our services.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the main function to display the homepage
main()