import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document, SearchHistory, DATABASE_URL

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

model_name = 'distiluse-base-multilingual-cased'
model = SentenceTransformer(model_name)

def embed_text(text):
    return model.encode([text])[0]  # Ensure the output is a 2D array for a single text

def search(query, top_n=5):
    print(f'Searching for query: {query}')
    query_embedding = embed_text(query)
    documents = session.query(Document).all()
    if not documents:
        print('No documents found in the database.')
        return []
    similarities = {}
    for doc in documents:
        doc_embedding = np.frombuffer(doc.embedding, dtype=np.float32)
        similarity = cosine_similarity([query_embedding], [doc_embedding]).item()
        similarities[doc.id] = similarity
    if not similarities:
        print('No similarities found.')
        return []
    sorted_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]
    top_docs = [(session.query(Document).filter_by(id=doc_id).first(), sim) for doc_id, sim in sorted_docs]
    for doc, sim in top_docs:
        print(f'Document: {doc.path} with similarity: {sim}')
        # Record search history
        search_history = SearchHistory(query=query, document_id=doc.id, timestamp=datetime.now())
        session.add(search_history)
    session.commit()
    return top_docs

if __name__ == '__main__':
    query = "Your query here"
    results = search(query, top_n=3)
    if results:
        for result in results:
            print(f'Document: {result[0].path}, Similarity: {result[1]}')
    else:
        print("No matching documents found.")
