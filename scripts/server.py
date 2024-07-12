from flask import Flask, request, render_template
import logging
from search import search
from qa import answer_question
from extract_text import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.form['query']
    logging.info(f'Received search query: {query}')
    top_doc_path = search(query)
    if not top_doc_path:
        return render_template('search.html', no_results=True)
    context = extract_text(top_doc_path)
    answer = answer_question(query, context)
    return render_template('search.html', results={'document_path': top_doc_path, 'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
