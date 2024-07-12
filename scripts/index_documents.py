import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from .extract_text import extract_text
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
            text = extract_text(file_path)
            if text:
                embedding = embed_text(text)
                doc = Document(path=file_path, embedding=embedding.tobytes())
                session.add(doc)
    session.commit()

if __name__ == '__main__':
    data_directory = '/home/solomon/data/lose_data/documents'
    index_documents(data_directory)
