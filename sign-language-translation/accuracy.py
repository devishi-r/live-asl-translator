from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2
import os

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to DB
conn = psycopg2.connect(
    database="postgres",
    host="localhost",
    user="postgres",
    password="os.getenv('POSTGRES_PASSWORD')",
    port=5432,
)
cur = conn.cursor()

# Step 1: Retrieve all words and embeddings
cur.execute("SELECT word, embedding FROM signs")
rows = cur.fetchall()

# Step 2: Create embedding dictionary
embedding_dict = {word: np.array(embedding) for word, embedding in rows}

# Step 3: Evaluation loop
correct = 0
total = 0
words = list(embedding_dict.keys())

for word in words:
    query_embedding = model.encode(word).reshape(1, -1)
    
    # Compare against all embeddings
    all_embeddings = np.array(list(embedding_dict.values()))
    similarities = cosine_similarity(query_embedding, all_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    predicted_word = words[best_match_idx]

    if predicted_word == word:
        correct += 1
    total += 1

print("Top-1 accuracy: {:.2f}%".format(100 * correct / total))
