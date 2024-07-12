import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

from scripts.index_documents import index_documents
from scripts.server import app

if __name__ == '__main__':
    data_directory = '/home/solomon/data/lose_data/documents'
    
    # Check if the database already exists
    db_path = '/home/solomon/data/lose_data/database.db'
    if not os.path.exists(db_path):
        print("Indexing documents...")
        index_documents(data_directory)
        print("Indexing completed.")
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
