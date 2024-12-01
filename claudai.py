import streamlit as st
import nltk
import numpy as np
import faiss
import redis
import pickle
from typing import List, Dict, Any

# Efficient Imports
import pymongo
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

class EfficientStudyAI:
    def __init__(self, 
                 mongodb_uri="mongodb://localhost:27017/", 
                 redis_host='localhost', 
                 redis_port=6379):
        # Optimized Database Connections
        self.mongo_client = pymongo.MongoClient(mongodb_uri, 
                                                maxPoolSize=10,
                                                socketTimeoutMS=3000,
                                                connectTimeoutMS=3000)
        self.db = self.mongo_client['studyai']
        
        # Redis Caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        
        # Lightweight Models
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_model = pipeline("question-answering", 
                                 model="distilbert-base-uncased-distilled-squad")
        
        # TF-IDF Vectorizer for additional similarity
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    def cached_embedding(self, question: str):
        """Efficient embedding with Redis caching"""
        cache_key = f"embedding:{question}"
        cached = self.redis_client.get(cache_key)
        
        if cached:
            return pickle.loads(cached)
        
        embedding = self.similarity_model.encode([question])[0]
        self.redis_client.set(cache_key, pickle.dumps(embedding))
        return embedding
    
    def find_similar_questions(self, 
                                available_time: float, 
                                similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Efficient question similarity search with FAISS"""
        # Fetch questions with indexing
        previous_papers = list(self.db['python'].find({"paper_type": "pre"}).hint([("paper_type", 1)]))
        model_papers = list(self.db['python'].find({"paper_type": "mod"}).hint([("paper_type", 1)]))
        
        previous_questions = [(q['_id'], q['question'], q['answer'], q['est_tim']) for q in previous_papers]
        model_questions = [(q['_id'], q['question'], q['answer']) for q in model_papers]
        
        # Efficient embedding
        pre_embeddings = np.array([self.cached_embedding(q[1]) for q in previous_questions])
        mod_embeddings = np.array([self.cached_embedding(q[1]) for q in model_questions])
        
        # FAISS for fast similarity search
        index = faiss.IndexFlatL2(pre_embeddings.shape[1])
        index.add(pre_embeddings)
        
        repeated_questions = []
        for j, (mod_id, mod_question, mod_answer) in enumerate(model_questions):
            query_embedding = mod_embeddings[j].reshape(1, -1)
            distances, indices = index.search(query_embedding, 1)
            
            if distances[0][0] < similarity_threshold:
                pre_idx = indices[0][0]
                pre_id, pre_question, pre_answer, pre_time = previous_questions[pre_idx]
                
                repeated_questions.append({
                    "pre_id": pre_id,
                    "pre_question": pre_question,
                    "pre_answer": pre_answer,
                    "mod_id": mod_id,
                    "mod_question": mod_question,
                    "mod_answer": mod_answer,
                    "similarity_score": 1 / (1 + distances[0][0]),
                    "est_time": pre_time
                })
        
        # Time-constrained selection
        selected_questions = []
        total_time = 0
        for repeated in sorted(repeated_questions, key=lambda x: (-x['similarity_score'], x['mod_id'])):
            if total_time + repeated['est_time'] <= available_time:
                selected_questions.append(repeated)
                total_time += repeated['est_time']
        
        return selected_questions
    
    def generate_contextual_answer(self, question: str, contexts: str) -> str:
        """Advanced question-answering with chunking"""
        # Sentence-based context chunking
        sentences = nltk.sent_tokenize(contexts)
        chunks = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
        
        # Parallel answer generation
        answers = []
        for chunk in chunks:
            try:
                result = self.qa_model(question=question, context=chunk, 
                                       max_answer_length=150)
                answers.append(result['answer'])
            except Exception:
                continue
        
        return ' '.join(set(answers))  # Unique answers

def main():
    st.title("Efficient Study AI Dashboard")
    
    # Initialize StudyAI
    study_ai = EfficientStudyAI()
    
    # Time-based question selection
    available_time = st.number_input("Exam Prep Time (minutes):", 
                                     min_value=0.0, step=1.0, value=60.0)
    
    # Find and display similar questions
    similar_questions = study_ai.find_similar_questions(available_time)
    
    st.subheader("Recommended Questions")
    for q in similar_questions:
        st.write(f"Question: {q['pre_question']}")
        st.write(f"Similarity: {q['similarity_score']:.2f}")
        st.write(f"Estimated Time: {q['est_time']} mins")
        st.write("---")
    
    # Question answering
    st.subheader("AI Answer Generator")
    user_question = st.text_input("Ask a question:")
    
    if user_question:
        # Fetch context
        context_data = list(study_ai.db['answers'].find({"intent": "intent"}))
        contexts = ' '.join([c['question'] for c in context_data])
        
        # Generate answer
        answer = study_ai.generate_contextual_answer(user_question, contexts)
        st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()