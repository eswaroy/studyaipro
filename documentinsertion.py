from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['studyai']  
collection = db['python']  
document = {
    "_id":19,
    "question":'''''',
    "answer": '''''',
    "est_tim":20,
    "paper_type":"pre"#mod

}


collection.insert_one(document)

print("Document inserted successfully.")

