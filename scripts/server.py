from flask import Flask, request, jsonify
from search import search
from qa import answer_question
from extract_text import extract_text

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.json['query']
    top_doc_path = search(query)
    context = extract_text(top_doc_path)
    answer = answer_question(query, context)
    return jsonify({'document_path': top_doc_path, 'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
