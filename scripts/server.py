from flask import Flask, request, render_template
from .search import search
from .qa import answer_question

app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.form['query']
    print(f'Received search query: {query}')
    top_docs = search(query, top_n=3)
    if not top_docs:
        return render_template('search.html', no_results=True, query=query)
    results = []
    for doc, sim in top_docs:
        context = doc.content  # Use stored content
        answer = answer_question(query, context)
        print(f'Question: {query}')
        print(f'Document Path: {doc.path}')
        print(f'Answer: {answer}')
        results.append({'document_path': doc.path, 'answer': answer, 'similarity': sim})
    return render_template('search.html', results=results, query=query)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
