from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline('question-answering')

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
