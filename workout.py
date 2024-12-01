import streamlit as st
import nltk
import numpy as np
import faiss
import redis
import pickle
from typing import List, Dict, Any
import pymongo
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer,util
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
client = MongoClient("mongodb://localhost:27017/")
db = client['studyai']
collection = db['python']
model = SentenceTransformer('bert-base-nli-mean-tokens')

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
                                     min_value=0.0, step=1.0, value=0.0)
    
    # Find and display similar questions
    similar_questions = study_ai.find_similar_questions(available_time)
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