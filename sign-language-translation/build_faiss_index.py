from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
import json
import faiss
import os
from dotenv import load_dotenv
load_dotenv()

# Step 1: Connect to PostgreSQL
conn = psycopg2.connect(
    database="postgres",
    host="localhost",
    user="postgres",
    password=os.getenv('POSTGRES_PASSWORD'),
    port=5432
)
cursor = conn.cursor()

# Step 2: Load all words from DB
cursor.execute("SELECT word FROM signs")
rows = cursor.fetchall()
words = [row[0] for row in rows]  # ['suit', 'tree', 'tail', ...]

# Step 3: Encode words using SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(words).astype('float32')  # shape: (num_words, 384)

# Step 4: Save words to words.json
with open("words.json", "w") as f:
    json.dump(words, f)

# Step 5: Save embeddings to .npy
np.save("word_embeddings.npy", embeddings)

# Step 6: Build and save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "word_index.faiss")

print("✅ FAISS index built and files saved:")
print("- words.json")
print("- word_embeddings.npy")
print("- word_index.faiss")
