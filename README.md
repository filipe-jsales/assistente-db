
fluxo inicial:

apagar o index faiss
rodar o generate embeddings
rodar o backend


python main.py > http://127.0.0.1:5000
$ streamlit run chat_app.py > http://localhost:8501

para utilizar fora do localhost, suba um tunneling do ngrok e troque o endpoint
http://localhost:5000/chat -> seudominiongrok.com/chat
