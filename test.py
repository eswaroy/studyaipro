import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
# Initialize MongoDB and model
client = MongoClient("mongodb://localhost:27017/")
db = client['studyai']
collection = db['python']
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Streamlit dashboard title and description
st.title("Study AI Question Repetition Dashboard")
st.write("This dashboard shows repeated questions based on semantic similarity.")

# Input: Available time
available_time = st.number_input("Enter the available time in minutes for the exam:", min_value=0.0, step=1.0)

# Fetch questions
previous_papers = collection.find({"paper_type": "pre"})
model_papers = collection.find({"paper_type": "mod"})

previous_questions = [(q['_id'], q['question'], q['answer'], q['est_tim']) for q in previous_papers]
model_questions = [(q['_id'], q['question'], q['answer']) for q in model_papers]

# Encode questions
if previous_questions and model_questions:
    pre_embeddings = model.encode([q[1] for q in previous_questions])
    mod_embeddings = model.encode([q[1] for q in model_questions])
    similarity_scores = util.pytorch_cos_sim(pre_embeddings, mod_embeddings)
    
    # Threshold and repeated questions
    similarity_threshold = 0.75
    repeated_questions = []

    for i, (pre_id, pre_question, pre_answer, pre_time) in enumerate(previous_questions):
        for j, (mod_id, mod_question, mod_answer) in enumerate(model_questions):
            score = similarity_scores[i][j].item()
            if score > similarity_threshold:
                repeated_questions.append({
                    "pre_id": pre_id,
                    "pre_question": pre_question,
                    "pre_answer": pre_answer,
                    "mod_id": mod_id,
                    "mod_question": mod_question,
                    "mod_answer": mod_answer,
                    "similarity_score": score,
                    "est_time": pre_time
                })

    # Display repeated questions based on available time
    sorted_repeated_questions = sorted(repeated_questions, key=lambda x: (-x['similarity_score'], x['mod_id']))
    selected_questions = []
    total_time = 0

    for repeated in sorted_repeated_questions:
        question_time = repeated['est_time']
        if total_time + question_time <= available_time:
            selected_questions.append(repeated)
            total_time += question_time

    # Output results in Streamlit
    st.subheader("Selected Questions Based on Available Time and Similarity")
    for repeated in selected_questions:
        st.write(f"**Repeated Question Found (Similarity: {repeated['similarity_score']:.2f})**")
        st.write(f"- **Previous Question:** {repeated['pre_question']}")
        st.write(f"- **Model Question:** {repeated['mod_question']}")
        st.write(f"- **Answer:** {repeated['mod_answer']}")
        st.write(f"- **Estimated time:** {repeated['est_time']} minutes")
        st.write("---")

else:
    st.warning("No questions were retrieved. Please check your MongoDB connection or data availability.")
st.write("QUESTIONS SHOULD LOOK IF THIS COMPLETED BEFORE TIME")
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


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
    st.write(f"Question: {question}\nAnswer: {answer}\n")
