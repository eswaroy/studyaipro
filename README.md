# STUDYAIPRO

STUDYAIPRO is an intelligent study assistant designed to streamline your exam preparation. By leveraging advanced AI models, the tool identifies relevant questions, recommends similar ones, and generates detailed answers to user queries using a blend of natural language processing and database integrations.

## Features

- **Efficient Question Recommendation**: Finds similar questions from a database of previous and model exam papers using BERT-based semantic similarity.
- **Time-Conscious Question Selection**: Recommends questions that can be completed within the user-defined available time.
- **Advanced Answer Generation**: Utilizes a question-answering model to provide contextual and accurate answers to user queries.
- **Optimized Database Integration**: Integrates with MongoDB and Redis for efficient data retrieval and caching.
- **Interactive Dashboard**: Built with Streamlit for a user-friendly interface.

## How It Works

1. **Similarity Search**:
   - Encodes questions from the database using BERT embeddings.
   - Matches previous exam questions with model exam questions based on semantic similarity.
   - Selects and ranks questions using a similarity threshold and estimated completion time.

2. **Answer Generation**:
   - Splits large contexts into smaller chunks.
   - Uses a question-answering pipeline to generate concise answers.
   - Combines and displays unique answers to the user's query.

3. **Caching and Optimization**:
   - Embeddings are cached using Redis to reduce computation time.
   - MongoDB serves as the primary database for storing and retrieving questions and answers.

## Prerequisites

Ensure the following are installed and running:

- Python 3.8+
- MongoDB
- Redis
- Streamlit
- Required Python libraries (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/STUDYAIPRO.git
   cd STUDYAIPRO
## filenames(desc)
documentinsertion.py---> for inserting the model and question paper questions and answers.
que-gen(db).py-----> for ai answer generation.
above both are related to database codes.
main.py------------> is the actual studyaipro code.
