from flask import Flask, request, jsonify
import torch
from search import search
from qa import answer_question
from extract_text import extract_text

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.json['query']
    top_doc_id = search(query, indexed_docs)
    context = extract_text(documents[top_doc_id])
    answer = answer_question(query, context)
    return jsonify({'document_id': top_doc_id, 'answer': answer})

if __name__ == '__main__':
    documents = {
        'doc1': '../data/pdfs/doc1.pdf',
        'doc2': '../data/pdfs/doc2.docx',
        'doc3': '../data/pdfs/doc3.txt',
        'doc4': '../data/pdfs/doc4.jpg'
    }
    indexed_docs = torch.load('../data/indexed_docs.pt')
    app.run(host='0.0.0.0', port=5000)
