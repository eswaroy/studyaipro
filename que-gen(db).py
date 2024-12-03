from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['studyai']  
collection = db['prompts']#questions  
document = {
    "_id":5,#u start from 20 in des order
    "question": '''explain about strings''',
    "answer":'''Though not commonly thought of as a "data structure," strings in Python can be treated as immutable sequences of characters.
Key Characteristics:
Immutable: Once created, strings cannot be altered.
Indexed: Each character in a string can be accessed using an index.
Usage Examples:
Processing text data, such as parsing and formatting.
Storing user input or output messages.
Real-World Application: Strings are central to data science and programming, especially in text analysis, web development, and API interactions.
        '''

}


collection.insert_one(document)

print("Document inserted successfully.")