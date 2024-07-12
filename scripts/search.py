import torch
import numpy as np
import logging
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from .index_documents import Document, DATABASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message)s')

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

model_name = 'xlm-roberta-base'
cache_dir = '/home/solomon/data/lose_data/models'

def embed_text(text):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = XLMRobertaModel.from_pretrained(model_name, cache_dir=cache_dir)
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def search(query):
    logging.info(f'Searching for query: {query}')
    query_embedding = embed_text(query)
    documents = session.query(Document).all()
    if not documents:
        logging.info('No documents found in the database.')
        return None
    similarities = {}
    for doc in documents:
        doc_embedding = np.frombuffer(doc.embedding, dtype=np.float32).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, doc_embedding).item()
        similarities[doc.id] = similarity
    if not similarities:
        logging.info('No similarities found.')
        return None
    top_doc_id = max(similarities, key=similarities.get)
    top_doc = session.query(Document).filter_by(id=top_doc_id).first()
    logging.info(f'Top document found: {top_doc.path}')
    logging.info(f'Similarities: {similarities}')
    return top_doc.path

if __name__ == '__main__':
    query = "Your query here"
