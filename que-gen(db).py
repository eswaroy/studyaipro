from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['studyai']  
collection = db['prompts']#questions  
document = {
    "_id":2,#u start from 20 in des order
    "question": '''explain about tuple''',
    "answer":'''A tuple is one of the built-in data structures in Python. It is used to store an ordered collection of items, much like a list. However, unlike lists, tuples are immutable, meaning their elements cannot be changed, added, or removed after creation. Tuples are ideal for situations where you want a collection of items that should remain constant throughout the program.
Tuples are ordered, immutable collections. Once created, their elements cannot be changed.
They are often used to store data that should not be modified, such as configuration settings or grouped values.
Ordered: Tuples maintain the order of the items as they were defined. The first item in the tuple will always have an index of 0, the second an index of 1, and so on.

Immutable: Once a tuple is created, you cannot modify its content. This makes tuples more efficient in terms of memory and performance compared to lists.

Heterogeneous: A tuple can hold items of different data types (e.g., integers, strings, floats, objects, etc.).

Indexed: You can access individual elements of a tuple using their indices, just like with lists.

Hashable: Tuples can be used as keys in dictionaries because they are immutable and hashable (provided all elements within the tuple are hashable).
Key Operations:
Access elements using an index (like lists).
Use slicing to retrieve subsets of elements.
Store heterogeneous data (e.g., (int, str, float))'''

}


collection.insert_one(document)

print("Document inserted successfully.")