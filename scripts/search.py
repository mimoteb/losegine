import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document, SearchHistory, DATABASE_URL
from .qa import answer_question

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

model_name = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def search(query, top_n=5):
    print(f'Searching for query: {query}')
    query_embedding = embed_text(query)
    documents = session.query(Document).all()
    if not documents:
        print('No documents found in the database.')
        return []
    similarities = []
    for doc in documents:
        doc_embedding = np.frombuffer(doc.embedding, dtype=np.float32)
        if doc_embedding.shape[0] != query_embedding.shape[0]:
            print(f'Dimension mismatch: {doc_embedding.shape[0]} vs {query_embedding.shape[0]}')
            continue
        similarity = cosine_similarity([query_embedding], [doc_embedding]).item()
        similarities.append((doc, similarity))
    if not similarities:
        print('No similarities found.')
        return []
    sorted_docs = sorted(similarities, key=lambda item: item[1], reverse=True)[:top_n]
    results = []
    for doc, sim in sorted_docs:
        print(f'Document: {doc.path} with similarity: {sim}')
        # Provide more context by using larger parts of the document content
        answer = answer_question(query, doc.content)
        results.append({'document_path': doc.path, 'answer': answer, 'similarity': sim})
        search_history = SearchHistory(query=query, document_id=doc.id, timestamp=datetime.now())
        session.add(search_history)
    session.commit()
    return results

if __name__ == '__main__':
    query = "Your query here"
    results = search(query, top_n=3)
    if results:
        for result in results:
            print(f'Document: {result["document_path"]}, Answer: {result["answer"]}, Similarity: {result["similarity"]}')
    else:
        print("No matching documents found.")
