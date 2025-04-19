from mistralai import Mistral
import os
from dotenv import load_dotenv
import os.path

def create_data_directory():
    """Create a data directory if it doesn't exist"""
    if not os.path.exists('data'):
        os.makedirs('data')

def extract_pdf_text(pdf_path):
    """Extract text from PDF and save each page to a separate file"""
    # Load environment variables
    load_dotenv()
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    # Upload the PDF file
    print(f"Uploading PDF file: {pdf_path}")
    uploaded_pdf = client.files.upload(
        file={
            "file_name": os.path.basename(pdf_path),
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )

    # Get a signed URL for the uploaded file
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    # Process the uploaded PDF file
    print("Processing PDF with OCR...")
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    # Create data directory if it doesn't exist
    create_data_directory()

    # Save each page's text content to a separate file
    print("Saving text files...")
    for i, page in enumerate(ocr_response.pages, 1):
        output_file = f"data/text_{i}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(page.markdown)
        print(f"Saved page {i} to {output_file}")

    print("Text extraction complete!")

if __name__ == "__main__":
    pdf_file = "ltimindtree_annual_report.pdf"
    if os.path.exists(pdf_file):
        extract_pdf_text(pdf_file)
    else:
        print(f"Error: {pdf_file} not found!")