def main():
    st.title("STUDYAIPRO")
    
    # Initialize StudyAI
    study_ai = EfficientStudyAI()
    
    # Pass/Fail Prediction Feature
    st.subheader("Exam Pass Prediction")
    attendance = st.number_input("Attendance Percentage:", min_value=0.0, max_value=100.0, step=0.1, value=75.0)
    study_hours = st.number_input("Study Hours per Week:", min_value=0.0, step=0.1, value=10.0)
    midterm1_score = st.number_input("Midterm 1 Score (%):", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
    midterm2_score = st.number_input("Midterm 2 Score (%):", min_value=0.0, max_value=100.0, step=0.1, value=50.0)
    Assignment_Completion=st.number_input("Assignment Completion (%):", min_value=0.0, max_value=100.0, step=0.1, value=60.0)

    
    if st.button("Predict Pass/Fail"):
        result = predict_pass_fail(attendance, study_hours, midterm1_score, midterm2_score,Assignment_Completion)
        st.write(f"**Prediction:** You are likely to **{result}** the exam.")
    
    # Existing functionalities
    available_time = st.number_input("Exam Prep Time (minutes):", min_value=0.0, step=1.0, value=0.0)
    similar_questions = study_ai.find_similar_questions(available_time)

    # Question answering
    st.subheader("AI Answer Generator")
    user_question = st.text_input("Ask a question:")
    
    if user_question:
        # Fetch context
        context_data = list(study_ai.db['prompts'].find({"question": user_question}))
        contexts = ' '.join([c['answer'] for c in context_data if 'answer' in c])
        
        # Generate answer
        answer = study_ai.generate_contextual_answer(user_question, contexts)
        st.write(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()