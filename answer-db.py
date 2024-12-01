from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['studyai']  
collection = db['answers']#questions  
document = {
    "_id":11,#u start from 20 in des order
    "question": '''Graphs are abstract data structures used to represent relationships between objects. They consist of nodes (or vertices) connected by edges. Graphs can be directed or undirected, weighted or unweighted. In Python, graphs are implemented using adjacency lists (via dictionaries) or specialized libraries like NetworkX. They are widely used in social networks, recommendation systems, and optimization problems. Common algorithms applied to graphs include depth-first search (DFS), breadth-first search (BFS), Dijkstra’s algorithm, and more. Python’s dynamic nature makes it a popular choice for implementing and analyzing graphs.
    Graphs can be represented using adjacency lists or matrices.
    Python uses dictionaries or libraries like NetworkX for graph manipulation.
    Supports algorithms like BFS, DFS, Dijkstra, etc.
    python
    Copy code
    graph = {"A": ["B", "C"], "B": ["A", "D"], "C": ["A"], "D": ["B"]}''',
    "context_type":"graph",
    "intent":"intent" #command for question
    #"intent":"intent" for answer or context
    
}


collection.insert_one(document)

print("Document inserted successfully.")