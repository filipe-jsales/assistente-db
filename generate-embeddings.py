import os
import requests
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()
embedding_dimension = 4096
index = faiss.IndexFlatL2(embedding_dimension)
context_file_path = 'navathe.txt'
faiss_index_path = 'faiss_index.index'
def generate_embeddings(texts, model="llama3.1:latest"):
    url = os.getenv("API_EMBEDDING_URL", "http://localhost:11434/api/embed")
    data = {
        "model": model,
        "input": texts
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json().get("embeddings", [])

def save_index():
    faiss.write_index(index, faiss_index_path)
    print("Índice FAISS salvo em", faiss_index_path)

def load_index():
    global index
    if os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        print("Índice FAISS carregado de", faiss_index_path)
    else:
        print("Nenhum índice FAISS encontrado. Criando um novo.")

def store_embeddings_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = [line.strip() for line in file if line.strip()]

    embeddings = generate_embeddings(texts)

    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)

    print(f"{len(embeddings)} embeddings foram adicionados ao índice FAISS.")
    save_index()

if not os.path.exists(faiss_index_path):
    store_embeddings_from_file(context_file_path)
else:
    load_index()