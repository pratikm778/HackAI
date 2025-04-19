from mistralai import Mistral
import os
from dotenv import load_dotenv
import base64

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

uploaded_pdf = client.files.upload(
    file={
        "file_name": "ltimindtree_annual_report.pdf",
        "content": open("ltimindtree_annual_report.pdf", "rb"),
    },
    purpose="ocr"
)  

print(uploaded_pdf)

# Get a signed URL for the uploaded file
signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

# Process the uploaded PDF file
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": signed_url.url,
    },
    include_image_base64=True
)

# Function to export images if needed
def data_uri_to_bytes(data_uri):
    _, encoded = data_uri.split(",", 1)
    return base64.b64decode(encoded)

def export_image(image):
    try:
        parsed_image = data_uri_to_bytes(image.image_base64)
        with open(f"image_{image.id}.png", "wb") as file:
            file.write(parsed_image)
    except Exception as e:
        print(f"Error exporting image: {e}")

# Write the markdown content to a file with UTF-8 encoding
with open('output.md', 'w', encoding='utf-8') as f:
    for page in ocr_response.pages:
        f.write(page.markdown)
        # Uncomment the following if you want to export images
        # for image in page.images:
        #     export_image(image)

print("OCR processing complete. Results saved to output.md")
