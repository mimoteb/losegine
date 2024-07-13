from transformers import AutoTokenizer, AutoModel
import torch
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from .models import Document, DATABASE_URL
import numpy as np

engine = create_engine(DATABASE_URL)
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

def update_document_embeddings():
    documents = session.query(Document).all()
    for doc in documents:
        doc_embedding = embed_text(doc.content, doc.path)
        doc.embedding = np.array(doc_embedding).tobytes()
        session.add(doc)
    session.commit()

if __name__ == '__main__':
    update_document_embeddings()
    print("Document embeddings updated successfully.")
