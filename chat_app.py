import streamlit as st
import requests

st.title("Assistente de Banco de Dados")
st.write("Faça suas perguntas sobre Banco de Dados!")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("Você:", "")

if user_input:
    response = requests.post("http://localhost:5000/chat", json={"input": user_input}).json()
    print('response:', response)
    assistant_response = response.get("response", "Erro ao conectar com a API.")

    st.session_state["history"].append({"user": user_input, "assistant": assistant_response})

for chat in st.session_state["history"]:
    st.write(f"Você: {chat['user']}")
    st.write(f"Assistente: {chat['assistant']}")
