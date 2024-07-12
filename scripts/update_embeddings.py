from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Document, DATABASE_URL
import numpy as np

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
model = SentenceTransformer(model_name)

def embed_text(text):
    return model.encode([text])[0]

def update_document_embeddings():
    documents = session.query(Document).all()
    for doc in documents:
        doc_embedding = embed_text(doc.content)
        doc.embedding = np.array(doc_embedding).tobytes()
        session.add(doc)
    session.commit()

if __name__ == '__main__':
    update_document_embeddings()
    print("Document embeddings updated successfully.")
