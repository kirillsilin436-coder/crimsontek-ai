import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Настройки страницы
st.set_page_config(page_title="CRIMSONTEK AI", page_icon="🔴", layout="wide")

# Дизайн
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    h1 {color: #ff4b4b;}
</style>
""", unsafe_allow_html=True)

st.title("🔴 CRIMSONTEK: Industrial Analyst")
st.caption("System v1.0")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    st.write("Status: 🟢 Online")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Загрузка файла
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        st.info("System process: Analyzing...")

        try:
            # Чтение и анализ
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(texts, embeddings)

            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4o-mini"),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            st.success("Ready.")

            # Чат
            query = st.chat_input("Enter command...")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.write(query)

                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        res = qa.invoke(query)
                        st.write(res['result'])
                        st.session_state.messages.append({"role": "assistant", "content": res['result']})

        except Exception as e:
            st.error(f"Error: {e}")
