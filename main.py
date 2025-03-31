from flask import Flask, request, jsonify
import os
import requests
import numpy as np
import faiss
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
context_file_path = 'navathe.txt'
faiss_index_path = 'faiss_index.index'

MAX_AMOUNT_MESSAGES = int(os.getenv("MAX_AMOUNT_MESSAGES", 10))
index = faiss.read_index(faiss_index_path)

message_history = []

def generate_embedding(text, model="llama3.1:latest"):
    url = os.getenv("API_EMBEDDING_URL", "http://localhost:11434/api/embed")
    data = {"model": model, "input": [text]}
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json().get("embeddings", [])[0]

def list_documents():
    if os.path.exists(context_file_path):
        with open(context_file_path, 'r') as file:
            return [line.strip() for line in file]
    return []

def get_relevant_context(query_text, top_k=5):
    query_embedding = generate_embedding(query_text)
    query_np = np.array([query_embedding]).astype('float32')
    print("Número de vetores no índice FAISS:", index.ntotal)
    distances, indices = index.search(query_np, k=top_k)
    all_texts = list_documents()

    print("Índices retornados pelo FAISS:", indices)  # Debug
    print("Total de documentos:", len(all_texts))  # Debug

    if len(all_texts) == 0:
        return "O contexto está vazio. Verifique o arquivo 'navathe.txt'."

    # Filtrar índices válidos
    valid_indices = [i for i in indices[0] if i >= 0 and i < len(all_texts)]

    if not valid_indices:
        return "Nenhum contexto relevante foi encontrado para a sua pergunta."

    results = [all_texts[i] for i in valid_indices]
    return " ".join(results)

def get_api_response(prompt, context=""):
    maximum_characters = int(os.getenv("CHARACTERS_AMOUNT", 100))
    system_instructions = (
        "Você é um assistente PT-BR especializado em Banco de Dados e Sistemas Gerenciadores de Banco de dados. "
        "Na primeira interação, cumprimente o usuário de maneira educada. "
        "Responda estritamente no tema relacionado a banco de dados, SGBD e afins; "
        "caso contrário, informe que não pode responder.; "
        "Utilize o contexto apenas como uma ajuda para responder a pergunta do usuário. As informações são advindas do livro Navathe, Elmasri. Sistemas de Banco de Dados. 6ª edição. Pearson, 2011. "
        f"Sua resposta deve ser limitada a quantidade de caracteres: {maximum_characters}, e responda em PT-BR."
    )
    history = "\n".join([f"Usuário: {msg['user']}\nAssistente: {msg['assistant']}" for msg in message_history])
    full_prompt = f"{system_instructions}\n\n{history}\n\n{context}\n\nUsuário: {prompt}\nAssistente:" if context else f"{system_instructions}\n\n{history}\n\nUsuário: {prompt}\nAssistente:"
    print('system_instructions:', system_instructions)
    headers = {"Content-Type": "application/json"}
    data = {"model": "llama3.1:latest", "prompt": full_prompt, "stream": False}
    url = os.getenv("API_COMPLETION_URL", "http://localhost:11434/api/generate")
    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()
    
    assistant_response = response_data.get('response', "Resposta não encontrada")
    message_history.append({"user": prompt, "assistant": assistant_response})
    
    if len(message_history) > MAX_AMOUNT_MESSAGES:
        message_history.pop(0)

    return assistant_response

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')
    if not user_input:
        return jsonify({"error": "Input is required"}), 400
    user_input_with_instruction = f"{user_input} Me diga de qual seção do livro você retirou essa informação."

    try:
        context = get_relevant_context(user_input_with_instruction )
        print("Context Generated:", context)  # Debug
        response = get_api_response(user_input, context)
        print("Response from API:", response)  # Debug
        return jsonify({"response": response})
    except Exception as e:
        print("Error in /chat endpoint:", e)  # Debug
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000)
