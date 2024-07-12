from transformers import pipeline

# Specify the model explicitly and cache directory
model_name = 'distilbert-base-cased-distilled-squad'
cache_dir = '/home/solomon/data/lose_data/models'

# Load the QA pipeline
qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name, cache_dir=cache_dir)

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
