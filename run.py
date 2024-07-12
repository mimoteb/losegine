from scripts.index_documents import index_documents
from scripts.server import app
import os

if __name__ == '__main__':
    data_directory = '/home/solomon/data'
    
    # Check if the database already exists
    db_path = './data/database.db'
    if not os.path.exists(db_path):
        print("Indexing documents...")
        index_documents(data_directory)
        print("Indexing completed.")
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
