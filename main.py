import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Carregar o modelo de embeddings para medir similaridade
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Função para ler perguntas e respostas do arquivo .txt
def load_qa_from_txt(input_path):
    qa_pairs = []
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        pairs = content.split("\n\n")  # Cada pergunta/resposta separado por uma linha em branco

        for pair in pairs:
            lines = pair.splitlines()
            if len(lines) >= 2:
                question = lines[0].replace("Pergunta:", "").strip()
                answer = lines[1].replace("Resposta:", "").strip()
                qa_pairs.append((question, answer))

    return qa_pairs


# Função para encontrar a pergunta mais semelhante e retornar a resposta
def get_answer_from_txt(user_question, qa_pairs):
    question_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
    doc_questions = [q for q, a in qa_pairs]
    doc_embeddings = embedding_model.encode(doc_questions, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(question_embedding, doc_embeddings)
    best_match_idx = np.argmax(similarities)

    best_question, best_answer = qa_pairs[best_match_idx]
    return best_answer


# Configuração da interface do Streamlit
st.title("Assistente Virtual")

# Carregar perguntas e respostas
input_file = 'chatbot.txt'
qa_pairs = load_qa_from_txt(input_file)

# Inicializar histórico de conversas
if "history" not in st.session_state:
    st.session_state.history = []

# Input direto na interface de chat
user_question = st.chat_input("Digite sua pergunta aqui...")

# Processar a pergunta quando o usuário envia
if user_question:
    answer = get_answer_from_txt(user_question, qa_pairs)
    st.session_state.history.append({"user": user_question, "bot": answer})

# Exibir histórico de mensagens no estilo chat com avatares personalizados
for chat in st.session_state.history:
    st.chat_message("user").markdown(chat["user"])  # Mensagem do usuário
    st.chat_message("assistant").markdown(chat["bot"])  # Resposta do assistente
