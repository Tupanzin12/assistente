import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import random

# Carregar o modelo de embeddings para medir similaridade
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Função para ler perguntas e respostas do arquivo JSON
def load_intents_from_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    return intents["intents"]


# Função para encontrar a melhor resposta usando similaridade de embeddings
def get_response_from_json(user_question, intents):
    user_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
    max_similarity = 0
    best_response = "Desculpe, não entendi. Pode reformular?"

    # Verificar cada padrão de pergunta em cada intenção
    for intent in intents:
        for pattern in intent["patterns"]:
            pattern_embedding = embedding_model.encode(pattern, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(user_embedding, pattern_embedding).item()

            # Se a similaridade for maior, salva a melhor resposta correspondente
            if similarity > max_similarity:
                max_similarity = similarity
                best_response = random.choice(intent["responses"])

    return best_response


# Configuração da interface do Streamlit
st.title("Assistente Virtual")

# Carregar perguntas e respostas do arquivo JSON
input_file = 'intents.json'
intents = load_intents_from_json(input_file)

# Inicializar histórico de conversas
if "history" not in st.session_state:
    st.session_state.history = []

# Input direto na interface de chat
user_question = st.chat_input("Digite sua pergunta aqui...")

# Processar a pergunta quando o usuário envia
if user_question:
    answer = get_response_from_json(user_question, intents)
    st.session_state.history.append({"user": user_question, "bot": answer})

# Exibir histórico de mensagens no estilo chat com avatares personalizados
for chat in st.session_state.history:
    st.chat_message("user").markdown(chat["user"])  # Mensagem do usuário
    st.chat_message("assistant").markdown(chat["bot"])  # Resposta do assistente
