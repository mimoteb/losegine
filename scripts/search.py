import torch
import numpy as np
import logging
from transformers import DistilBertTokenizer, DistilBertModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from scripts.index_documents import Document, DATABASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def embed_text(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='/home/solomon/data/lose_data/models')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir='/home/solomon/data/lose_data/models')
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
    similarities = {doc.id: cosine_similarity(query_embedding, np.frombuffer(doc.embedding, dtype=np.float32).reshape(1, -1)).item() for doc in documents}
    if not similarities:
        logging.info('No similarities found.')
        return None
    top_doc_id = max(similarities, key=similarities.get)
    top_doc = session.query(Document).filter_by(id=top_doc_id).first()
    logging.info(f'Top document found: {top_doc.path}')
    return top_doc.path

if __name__ == '__main__':
    query = "Your query here"
    result = search(query)
    if result:
        print(result)
    else:
        print("No matching documents found.")
