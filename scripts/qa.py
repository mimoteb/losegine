from transformers import pipeline

model_name = 'deepset/roberta-base-squad2'
qa_pipeline = pipeline('question-answering', model=model_name)

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']
