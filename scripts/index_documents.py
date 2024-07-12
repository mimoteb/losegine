import torch
from transformers import DistilBertTokenizer, DistilBertModel
from extract_text import extract_text

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def index_documents(documents):
    return {doc_id: embed_text(extract_text(doc_path)) for doc_id, doc_path in documents.items()}

if __name__ == '__main__':
    documents = {
        'doc1': '../data/pdfs/doc1.pdf',
        'doc2': '../data/pdfs/doc2.docx',
        'doc3': '../data/pdfs/doc3.txt',
        'doc4': '../data/pdfs/doc4.jpg'
    }
    indexed_docs = index_documents(documents)
    torch.save(indexed_docs, '../data/indexed_docs.pt')
