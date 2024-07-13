import os
from transformers import AutoTokenizer, AutoModel
import torch
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .extract_text import extract_text

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True)
    title = Column(String)
    content = Column(Text)
    embedding = Column(LargeBinary)

DATABASE_PATH = '/home/solomon/data/lose_data/database.db'
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

model_name = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text, path):
    combined_text = f"{text} {path}"
    inputs = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def index_documents(directory):
    if not os.path.exists(directory):
        print(f'Directory {directory} does not exist.')
        return

    document_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            existing_doc = session.query(Document).filter_by(path=file_path).first()
            if existing_doc:
                print(f'Document already exists in the database: {file_path}')
                continue
            print(f'Reading file: {file_path}')
            text = extract_text(file_path)
            if text:
                print(f'Indexing file: {file_path}')
                embedding = embed_text(text, file_path)
                doc = Document(path=file_path, title=file, content=text, embedding=embedding.tobytes())
                session.add(doc)
                document_count += 1
            else:
                print(f'No text extracted from file: {file_path}')
    session.commit()
    print(f'Indexing completed. {document_count} documents indexed.')

    documents = session.query(Document).all()
    print(f'{len(documents)} documents found in the database.')
    for doc in documents:
        print(f'Document: {doc.path}')

if __name__ == '__main__':
    data_directory = '/home/solomon/data/lose_data/documents'
    print(f'Starting indexing for directory: {data_directory}')
    index_documents(data_directory)
