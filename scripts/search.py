import torch
from sklearn.metrics.pairwise import cosine_similarity
from index_documents import embed_text

def search(query, indexed_docs):
    query_embedding = embed_text(query)
    similarities = {doc_id: cosine_similarity(query_embedding, doc_embedding).item() for doc_id, doc_embedding in indexed_docs.items()}
    return max(similarities, key=similarities.get)

if __name__ == '__main__':
    indexed_docs = torch.load('../data/indexed_docs.pt')
    query = "Your query here"
    print(search(query, indexed_docs))
