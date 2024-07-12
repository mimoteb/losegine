import torch
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from index_documents import Document, DATABASE_URL

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def embed_text(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def search(query):
    query_embedding = embed_text(query)
    documents = session.query(Document).all()
    similarities = {doc.id: cosine_similarity(np.frombuffer(query_embedding, dtype=np.float32), 
                                              np.frombuffer(doc.embedding, dtype=np.float32).reshape(1, -1)).item() 
                    for doc in documents}
    top_doc_id = max(similarities, key=similarities.get)
    top_doc = session.query(Document).filter_by(id=top_doc_id).first()
    return top_doc.path

if __name__ == '__main__':
    query = "Your query here"
    print(search(query))
