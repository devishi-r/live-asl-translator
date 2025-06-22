from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
import json
import faiss
import os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    database="postgres",
    host="localhost",
    user="postgres",
    password=os.getenv('POSTGRES_PASSWORD'),
    port=5432
)
cursor = conn.cursor()

cursor.execute("SELECT word FROM signs")
rows = cursor.fetchall()
words = [row[0] for row in rows]  # ['suit', 'tree', 'tail', ...]

# encoding words into embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(words).astype('float32')  # shape: (num_words, 384)

with open("words.json", "w") as f:
    json.dump(words, f)

np.save("word_embeddings.npy", embeddings)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "word_index.faiss")

print("FAISS index built and files saved:")