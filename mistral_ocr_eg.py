from mistralai import Mistral
import os
from dotenv import load_dotenv

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

uploaded_pdf.
