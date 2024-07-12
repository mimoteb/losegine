from flask import Flask, request, jsonify, render_template_string
from search import search
from qa import answer_question
from extract_text import extract_text

app = Flask(__name__)
app.config['DEBUG'] = True

# HTML template for the search form and results
HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Local Search Engine</title>
  </head>
  <body>
    <div style="text-align: center; margin-top: 50px;">
      <h1>Local Search Engine</h1>
      <form action="/search" method="post">
        <input type="text" name="query" placeholder="Enter your query" style="width: 300px; padding: 10px;">
        <input type="submit" value="Search" style="padding: 10px;">
      </form>
      {% if results %}
        <h2>Results:</h2>
        <p><strong>Document Path:</strong> {{ results['document_path'] }}</p>
        <p><strong>Answer:</strong> {{ results['answer'] }}</p>
      {% elif no_results %}
        <h2>No matching documents found.</h2>
      {% endif %}
    </div>
  </body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.form['query']
    top_doc_path = search(query)
    if not top_doc_path:
        return render_template_string(HTML_TEMPLATE, no_results=True)
    context = extract_text(top_doc_path)
    answer = answer_question(query, context)
    return render_template_string(HTML_TEMPLATE, results={'document_path': top_doc_path, 'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
