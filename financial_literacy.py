import streamlit as st
from rag_model_for_finlit import RAGGenerator  # Import the RAGGenerator class

# Financial terms to display in the sidebar
financial_terms = [
    "Return on Equity (ROE) – 25.0% reported",
    "Profit After Tax (PAT) – INR 45,846 Million",
    "Basic and Diluted EPS (Earnings per Share) – INR 151.60 and INR 151.24 respectively",
    "Operating Margins (can be derived from revenue and expenses)",
    "Net Profit Margin",
    "Gross Margin",
    "EBITDA and EBITDA Margin (implicitly derived through P&L)",
    "Current Ratio – Derived from current assets and current liabilities",
    "Quick Ratio – More specific liquidity ratio using quick assets",
    "Debt to Equity Ratio",
    "Return on Assets (ROA)",
    "Asset Turnover Ratio",
]

# Initialize the RAG model
rag_model = RAGGenerator()  # Create an instance of the RAGGenerator class

# Streamlit app
def main():
    st.set_page_config(page_title="Financial Literacy", layout="wide")

    # Sidebar
    with st.sidebar:
        st.title("Learn Fundamentals Thru 10K")
        st.markdown("### Some example terms are listed below:")
        for term in financial_terms:
            st.markdown(f"- {term}")

    # Main page
    st.title("Financial Literacy Assistant")
    st.markdown("Ask any question about financial concepts or terms, and I'll help you understand them!")

    # Search bar
    query = st.text_input("Enter your question:", placeholder="e.g., What is Return on Equity (ROE)?")

    if query:
        with st.spinner("Fetching explanation..."):
            try:
                # Call the RAG model to get a response
                result = rag_model.generate_answer(query)
                answer = result.get("answer", "No answer available.")
                sources = result.get("sources", [])

                # Display the answer
                st.markdown("### Explanation:")
                st.write(answer)

                # Display sources if available
                if sources:
                    st.markdown("### Sources:")
                    for source in sources:
                        if source["type"] == "text":
                            st.markdown(f"- **Page {source['page']}**: {source['content_preview']}")
                        elif source["type"] == "image":
                            st.markdown(f"- **Page {source['page']}**: Image at {source['path']}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()