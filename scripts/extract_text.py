from pdfminer.high_level import extract_text as extract_text_from_pdf
import docx2txt
from PIL import Image
import pytesseract

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text

def extract_text(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    elif ext in ['png', 'jpg', 'jpeg']:
        return extract_text_from_image(file_path)
    else:
        return ""
