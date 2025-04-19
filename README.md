# Multimodal RAG System

A Retrieval-Augmented Generation (RAG) system that can process and query both text and images from corporate documents.

## Overview

This project implements a multimodal RAG system that:
1. Extracts and processes text and images from PDF documents
2. Embeds text and images into vector space using Google's embedding models
3. Stores embeddings in ChromaDB vector database
4. Retrieves relevant content based on user queries
5. Generates coherent responses using Mistral AI

## System Architecture

```
User Query → Multimodal Retriever → RAG Generator → User Response
                  ↑                        ↑
              ChromaDB ← Embeddings Processor
                  ↑
      Text Files & Images ← PDF Processor
                  ↑
          Source Documents
```

## Components

### PDF Processing & Embedding
- **pdf_processor.py**: Extracts text content from PDFs and splits into chunks
- **pic_record.py**: Extracts images from PDF files
- **embeddings_processor.py**: Creates embeddings for text using Google's embedding models

### Retrieval & Generation
- **multimodal_retriever.py**: Handles multimodal queries across text and image collections
- **rag_generator.py**: Takes retrieval results and generates responses using Mistral AI
- **retriever.py**: Simple text-only retrieval functions (legacy)

### User Interface & Evaluation
- **gradio_interface.py**: Web interface using Gradio for interacting with the system
- **evaluation.py**: Tools to evaluate system performance
- **optimization.py**: Parameter optimization to fine-tune the system

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)
- API keys:
  - Google AI API key (for embeddings)
  - Mistral AI API key (for LLM)

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your API keys:
```
GOOGLE_API_KEY=your_google_ai_key
MISTRAL_API_KEY=your_mistral_ai_key
```

### Usage

#### Processing Documents

Process a PDF document to extract text and images:

```bash
python pdf_processor.py --input yourfile.pdf
python pic_record.py --input yourfile.pdf
```

Generate embeddings and store in ChromaDB:

```bash
python embeddings_processor.py
```

#### Query the System

Launch the web interface:

```bash
python gradio_interface.py
```

Or use the system programmatically:

```python
from rag_generator import RAGGenerator

generator = RAGGenerator()
result = generator.generate_answer("What were the financial highlights from the last fiscal year?")
print(result['answer'])
```

#### Evaluate and Optimize

Run evaluation:

```bash
python evaluation.py
```

Fine-tune parameters:

```bash
python optimization.py
```

## System Optimization

For optimal performance, the system can be tuned with these parameters:
- Number of text chunks to retrieve (default: 5)
- Number of images to retrieve (default: 3)
- LLM temperature (default: 0.1)

Run the optimization script to find the best parameters for your specific data.

## Extending the System

### Adding New Document Types
- Implement new document processors that output text chunks and images
- Process these through the embedding pipeline

### Using Different Models
- Modify the embedding functions in `embeddings_processor.py`
- Change the LLM in `rag_generator.py`

## Limitations

- The system currently only handles PDF documents
- Image understanding is limited by the quality of embeddings
- Performance depends on the quality and relevance of the source documents

## Future Improvements

- Implement CLIP or other specialized multimodal embedding models
- Add support for more document types (Word, PPT, etc.)
- Implement user feedback loop for continuous improvement
- Add caching mechanisms for frequently asked questions

## License

This project is licensed under the MIT License - see the LICENSE file for details.
