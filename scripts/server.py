from flask import Flask, request, render_template
import logging
from .search import search
from .qa import answer_question
from .extract_text import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message=s')

app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.form['query']
    logging.info(f'Received search query: {query}')
    top_docs = search(query, top_n=3)  # Return top 3 results
    if not top_docs:
        return render_template('search.html', no_results=True)
    results = []
    for doc, sim in top_docs:
        context = extract_text(doc.path)
        answer = answer_question(query, context)
        logging.info(f'Question: {query}')
        logging.info(f'Document Path: {doc.path}')
        logging.info(f'Answer: {answer}')
        results.append({'document_path': doc.path, 'answer': answer, 'similarity': sim})
    return render_template('search.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
