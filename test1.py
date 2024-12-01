from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import streamlit as st
# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['studyai']
collection = db['python']

# Load pre-trained BERT model for semantic similarity
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

# MongoDB setup and question-answering pipeline remain the same
questions_collection = db['questions']
contexts_collection = db['answers']

model = SentenceTransformer('bert-base-nli-mean-tokens')
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Fetch questions and contexts from MongoDB
ques_retrieve = questions_collection.find({"command": "command"})
context_retrieve = contexts_collection.find({"intent": "intent"})

# Extract and process questions and contexts
ques = [(q['_id'], q['question']) for q in ques_retrieve]
context = [(c['_id'], c['question']) for c in context_retrieve]
ques_embeddings = model.encode([q[1] for q in ques])
context_texts = [c[1] for c in context]
contexts = " ".join(context_texts)

# Input question from the user
question_text = st.text_input("Type your question here...")

if question_text:
    question_embedding = model.encode([question_text])
    similarity_scores = util.pytorch_cos_sim(ques_embeddings, question_embedding)
    similarity_threshold = 0.70

    question_retrieval = []
    for i, score in enumerate(similarity_scores):
        if score.item() > similarity_threshold:
            question_retrieval.append({
                "answer": contexts,
                "question_db": ques[i][1]
            })

    context_list = [q["answer"] for q in question_retrieval]
    question_list = [q["question_db"] for q in question_retrieval]

    def context_generator(data, chunk_size=700):
        """Yield overlapping chunks of context."""
        for i in range(0, len(data), chunk_size):
            yield " ".join(data[i:i + chunk_size])
    def generate_long_answers(data, questions):
        aggregated_answers = []
        for question in questions:
            partial_answers = []
            for context_chunk in context_generator(data, chunk_size=50):
                result = qa_pipeline(question=question, context=context_chunk, min_length=800, max_length=1500)
                
                # Remove extra spaces from the answer
                formatted_answer = result['answer'].replace(" ", "")
                
                # Append formatted answer
                partial_answers.append(formatted_answer)
            
            # Combine all partial answers
            combined_answer = " ".join(partial_answers)
            aggregated_answers.append((question, combined_answer))
        
        return aggregated_answers

# Display results in a readable format
    final_result = generate_long_answers(contexts, question_list)
    for question, answer in final_result:
        st.write(f"**Question**: {question}")
        st.write(f"**Answer**: {answer}")
