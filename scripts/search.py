import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from .index_documents import Document, DATABASE_URL

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

def search(query, top_n=5):
    print(f'Searching for query: {query}')
    query_embedding = embed_text(query)
    documents = session.query(Document).all()
    if not documents:
        print('No documents found in the database.')
        return []
    similarities = {}
    for doc in documents:
        doc_embedding = np.frombuffer(doc.embedding, dtype=np.float32).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, doc_embedding).item()
        similarities[doc.id] = similarity
    if not similarities:
        print('No similarities found.')
        return []
    sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    top_docs = [(session.query(Document).filter_by(id=doc_id).first(), sim) for doc_id, sim in sorted_docs]
    for doc, sim in top_docs:
        print(f'Document: {doc.path} with similarity: {sim}')
    return top_docs

if __name__ == '__main__':
    query = "Your query here"
    results = search(query, top_n=3)
    if results:
        for result in results:
            print(f'Document: {result[0].path}, Similarity: {result[1]}')
    else:
        print("No matching documents found.")
