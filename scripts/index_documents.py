import os
import torch
import logging
from transformers import DistilBertTokenizer, DistilBertModel
from scripts.extract_text import extract_text
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True)
    embedding = Column(LargeBinary)

DATABASE_PATH = '/home/solomon/data/lose_data/database.db'
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='/home/solomon/data/lose_data/models')
model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir='/home/solomon/data/lose_data/models')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def index_documents(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            logging.info(f'Reading file: {file_path}')
            text = extract_text(file_path)
            if text:
                logging.info(f'Indexing file: {file_path}')
                embedding = embed_text(text)
                doc = Document(path=file_path, embedding=embedding.tobytes())
                session.add(doc)
    session.commit()
    logging.info('Indexing completed.')

if __name__ == '__main__':
    data_directory = '/home/solomon/data/lose_data/documents'
    logging.info(f'Starting indexing for directory: {data_directory}')
    index_documents(data_directory)
