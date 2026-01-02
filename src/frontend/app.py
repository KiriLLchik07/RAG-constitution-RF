import streamlit as st
import requests
from typing import Any

st.set_page_config(
    page_title="Конституция РФ - Вопросы и Ответы",
    page_icon="🇷🇺",
    layout="wide"
)

API_URL = "http://localhost:8000/ask"

st.title("🇷🇺 Конституция Российской Федерации")
st.markdown("### Задайте ваш вопрос, и я найду ответ в Конституции РФ")

if "messages" not in st.session_state:
    st.session_state.messages = []

def ask_question(question: str) -> dict[str, Any]:
    """Отправляет вопрос к FastAPI и возвращает ответ"""
    try:
        with st.spinner("Ищу ответ в Конституции РФ..."):
            response = requests.post(
                API_URL,
                json={
                    "question": question,
                    "n_initial": 10,
                    "n_final": 5
                },
                timeout=300
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Ошибка при запросе: {response.status_code}\n{response.text}",
                "sources": [],
                "execution_time": 0.0
            }
    except Exception as e:
        return {
            "answer": f"Произошла ошибка при подключении к серверу: {str(e)}",
            "sources": [],
            "execution_time": 0.0
        }

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**Вы:** {message['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Конституция РФ:** {message['content']}")
                
                if "sources" in message and message["sources"]:
                    with st.expander("Источники"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Статья {source['article_number']} ({source['chapter']})**")
                            st.markdown(f"Релевантность: {source['score']:.4f}")
                            st.markdown(f"*{source['text_excerpt']}*")
                            st.markdown("---")

if prompt := st.chat_input("Введите ваш вопрос о Конституции РФ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(f"**Вы:** {prompt}")
    
    result = ask_question(prompt)
    
    answer_text = result["answer"]
    sources = result.get("sources", [])
    execution_time = result.get("execution_time", 0.0)
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer_text,
        "sources": sources,
        "execution_time": execution_time
    })
    
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(f"**Конституция РФ:** {answer_text}")
            
            st.caption(f"Время обработки: {execution_time:.2f} сек")
            
            if sources:
                with st.expander("Источники"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Статья {source['article_number']} ({source['chapter']})**")
                        st.markdown(f"Релевантность: {source['score']:.4f}")
                        st.markdown(f"*{source['text_excerpt']}*")
                        st.markdown("---")
            else:
                st.warning("Не найдены релевантные источники в Конституции РФ")

with st.sidebar:
    st.header("О приложении")
    st.markdown("""
    Это приложение использует RAG-систему для поиска ответов на вопросы в тексте Конституции Российской Федерации.
    
    **Как это работает:**
    1. Вы задаете вопрос
    2. Система ищет релевантные статьи в Конституции
    3. Языковая модель генерирует ответ на основе найденных статей
    
    **Технологии:**
    - FastAPI (бэкенд)
    - Streamlit (фронтенд)
    - Ollama + Mistral (LLM)
    - ChromaDB (векторное хранилище)
    """)
    
    st.markdown("---")
    
    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Статус API:**")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            st.success("API работает!")
        else:
            st.error("API недоступен!")
    except requests.exceptions.ConnectionError:
        st.error("API недоступен!")

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatInput {
        position: fixed;
        bottom: 2rem;
        width: 90%;
        max-width: 1200px;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)
