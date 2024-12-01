from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['study']
collection = db['test']

# Load pre-trained BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Fetch previous year papers and model papers questions
previous_papers = collection.find({"paper": "pre"})
model_papers = collection.find({"paper": "mod"})

# Extract questions from the database
previous_questions = [(q['_id'], q['question'], q['answer'],q['est_time']) for q in previous_papers]
model_questions = [(q['_id'], q['question'], q['answer']) for q in model_papers]

# Encode the questions using BERT model
pre_embeddings = model.encode([q[1] for q in previous_questions])  # Encode previous paper questions
mod_embeddings = model.encode([q[1] for q in model_questions])     # Encode model paper questions

# Compare semantic similarity using cosine similarity
similarity_scores = util.pytorch_cos_sim(pre_embeddings, mod_embeddings)

# Set a threshold for similarity (you can adjust it)
similarity_threshold = 0.85

# Store repeated questions based on semantic similarity
repeated_questions = []

for i, (pre_id, pre_question, pre_answer,pre_time) in enumerate(previous_questions):
    for j, (mod_id, mod_question, mod_answer) in enumerate(model_questions):
        if similarity_scores[i][j].item() > similarity_threshold:
            repeated_questions.append({
                "pre_id": pre_id,
                "pre_question": pre_question,
                "pre_answer": pre_answer,
                "mod_id": mod_id,
                "mod_question": mod_question,
                "mod_answer": mod_answer,
                "similarity_score": similarity_scores[i][j].item(),
                "est_time":pre_time
            })

# Output the repeated questions
for repeated in repeated_questions:
    print(f"Repeated Question Found (Similarity: {repeated['similarity_score']:.2f}):")
    print(f" - M(QN): {repeated['mod_id']},P(QN): {repeated['pre_id']}, Pre Question: {repeated['pre_question']}, matched with MQ  Question: {repeated['mod_question']}\n\n Answer: {repeated['pre_answer']}\n\t\t\t\t\t\t\t\t\t\t Estimated time: {repeated['est_time']}")
