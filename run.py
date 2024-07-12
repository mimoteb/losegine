import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

from scripts.index_documents import index_documents
from scripts.server import app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    data_directory = '/home/solomon/data/lose_data/documents'
    
    # Force re-indexing for testing
    force_reindex = True  # Change this to False to disable re-indexing

    # Ensure the database path exists
    db_path = '/home/solomon/data/lose_data/database.db'
    if force_reindex or not os.path.exists(db_path):
        logging.info("Indexing documents...")
        index_documents(data_directory)
        logging.info("Indexing completed.")
    else:
        logging.info("Database already exists. Skipping indexing.")

    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
