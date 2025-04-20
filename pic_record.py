import os
import sys
try:
    import pymupdf as fitz
except ImportError:
    print("Please install PyMuPDF using: pip install PyMuPDF")
    sys.exit(1)
try:
    import pdfplumber
except ImportError:
    print("Please install pdfplumber using: pip install pdfplumber")
    sys.exit(1)
try:
    from pdf2image import convert_from_path
except ImportError:
    print("Please install pdf2image using: pip install pdf2image")
    sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm using: pip install tqdm")
    sys.exit(1)
from PIL import Image

# Define default PDF path
DEFAULT_PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltimindtree_annual_report.pdf")

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    img_count = 0
    
    # First count total images
    total_images = sum(len(page.get_images(full=True)) for page in doc)
    print(f"[*] Found {total_images} images to extract")
    
    progress_bar = tqdm(total=total_images, desc="Extracting images")
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            out_name = f"page{page_index+1}_img{img_index}.{image_ext}"
            with open(os.path.join(output_folder, out_name), "wb") as img_file:
                img_file.write(image_bytes)
            img_count += 1
            progress_bar.update(1)
    
    progress_bar.close()
    print(f"[+] Successfully extracted {img_count} images")

def extract_tables(pdf_path, output_folder, dpi=200):
    print("[*] Converting PDF pages to images...")
    pil_pages = convert_from_path(pdf_path, dpi=dpi)
    table_count = 0
    
    print("[*] Detecting tables...")
    with pdfplumber.open(pdf_path) as pdf:
        # First count total tables
        total_tables = sum(len(page.find_tables()) for page in pdf.pages)
        print(f"[*] Found {total_tables} tables to extract")
        
        progress_bar = tqdm(total=total_tables, desc="Extracting tables")
        
        for page_index, page in enumerate(pdf.pages):
            tables = page.find_tables()
            if not tables:
                continue

            pil_page = pil_pages[page_index]
            page_h = pil_page.height

            for tbl_index, table in enumerate(tables, start=1):
                x0, y0, x1, y1 = table.bbox
                top = page_h - y1
                bottom = page_h - y0
                left = x0
                right = x1

                crop = pil_page.crop((left, top, right, bottom))
                out_name = f"page{page_index+1}_table{tbl_index}.png"
                crop.save(os.path.join(output_folder, out_name))
                table_count += 1
                progress_bar.update(1)
        
        progress_bar.close()
    print(f"[+] Successfully extracted {table_count} tables")

def main():
    pdf_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_PDF_PATH
    
    if not os.path.isfile(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    # make output folder next to this script
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(base_dir, "pic_data")
    ensure_folder(output_dir)

    extract_images(pdf_path, output_dir)
    extract_tables(pdf_path, output_dir)

if __name__ == "__main__":
    main()
