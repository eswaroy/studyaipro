from transformers import pipeline
from pymongo import MongoClient

# Load the question-answering pipeline with a pre-trained model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['studyai']

# Fetch questions from the questions collection
questions_collection = db['questions']
questions = list(questions_collection.find({}, {"_id": 0, "question": 1}))

# Fetch contexts from the contexts collection
contexts_collection = db['answers']
answers = list(contexts_collection.find({}, {"_id": 0, "question": 1}))

# Prepare the data for answers generation
questions = [q['question'] for q in questions]
context_texts = [c['question'] for c in answers]

# Join contexts into a single string for processing
context = " ".join(context_texts)

# Function to chunk the context if it's too large
def context_generator(data, chunk_size=100):
    """Yield chunks of context from the large dataset."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Function to generate and aggregate answers from context
def generate_long_answers(data, questions):
    """Generate answers for a list of questions using the provided context."""
    aggregated_answers = []
    for question in questions:
        partial_answers = []
        for context_chunk in context_generator(data):
            # Generate answer from each context chunk
            result = qa_pipeline(question=question, context=context_chunk)
            partial_answers.append(result['answer'])
        
        # Concatenate all partial answers for a more comprehensive answer
        combined_answer = " ".join(partial_answers)
        aggregated_answers.append((question, combined_answer))
    return aggregated_answers

# Generate and print answers
answers = generate_long_answers(context, questions)
for question, answer in answers:
    print(f"Question: {question}\nAnswer: {answer}\n")
