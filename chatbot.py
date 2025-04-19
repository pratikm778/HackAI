import os
import subprocess
from flask import Flask, request, jsonify
from retriever import query_chroma_db
from embeddings_processor import EmbeddingsProcessor
from dotenv import load_dotenv
from transformers import pipeline

# Ensure required modules are installed
required_modules = ["openai", "transformers"]
for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call(["pip", "install", module])

# ✅ Move this import AFTER ensuring it's installed
import openai

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Initialize EmbeddingsProcessor
processor = EmbeddingsProcessor()

class DocumentChatbot:
    def __init__(self, document_path):
        self.document_path = document_path
        self.document_content = None
        self.qa_pipeline = pipeline("question-answering")

    def load_document(self):
        if not os.path.exists(self.document_path):
            raise FileNotFoundError(f"Document not found at {self.document_path}")
        with open(self.document_path, 'r', encoding='utf-8') as file:
            self.document_content = file.read()

    def answer_question(self, question):
        if not self.document_content:
            raise ValueError("Document not loaded. Please load the document first.")
        result = self.qa_pipeline(question=question, context=self.document_content)
        return result['answer']

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_query = data.get("query", "")
    top_n = data.get("top_n", 5)

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        results = processor.query_similar_content(user_query, n_results=top_n)
        context = "\n\n".join([result['text'] for result in results])

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
            ]
        )

        answer = response['choices'][0]['message']['content']
        return jsonify({"answer": answer, "context": context})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Example usage
    document_path = "/Users/nirmal/Desktop/HackAI/sample_document.txt"  # Replace with your document path
    chatbot = DocumentChatbot(document_path)

    try:
        chatbot.load_document()
        print("Document loaded successfully.")
        while True:
            question = input("Ask a question about the document (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            answer = chatbot.answer_question(question)
            print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")

    app.run(host="0.0.0.0", port=5000)
