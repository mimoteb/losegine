from pdfminer.high_level import extract_text as extract_text_from_pdf
import docx2txt
from PIL import Image
import pytesseract
import logging

def extract_text_from_docx(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        logging.error(f'Error extracting text from docx: {e}')
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f'Error extracting text from txt: {e}')
        return ""

def extract_text_from_image(file_path):
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    except pytesseract.TesseractNotFoundError as e:
        logging.error(f'Tesseract not found: {e}')
        return ""
    except Exception as e:
        logging.error(f'Error extracting text from image: {e}')
        return ""

def extract_text(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == 'pdf':
        try:
            return extract_text_from_pdf(file_path)
        except Exception as e:
            logging.error(f'Error extracting text from pdf: {e}')
            return ""
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    elif ext in ['png', 'jpg', 'jpeg']:
        return extract_text_from_image(file_path)
    else:
        logging.warning(f'Unsupported file type: {file_path}')
        return ""
