import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ---------------- UI ----------------
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("🤖 AI PDF Chat Assistant")

# ---------------- Session State ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Upload PDF ----------------
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file is not None:

    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in FAISS (save once)
    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("Document processed successfully!")

# ---------------- Chat Section ----------------
if st.session_state.vectorstore:

    query = st.chat_input("Ask something about your PDF...")

    if query:
        vectorstore = st.session_state.vectorstore

        # Retrieve relevant chunks
        results = vectorstore.similarity_search(query, k=3)

        context = "\n".join([doc.page_content for doc in results])

        # Load LLM
        llm = OllamaLLM(model="llama3")

        # Prompt
        prompt = f"""
        You are a helpful assistant.
        Answer ONLY from the context below.
        If answer is not found, say "I don't know".

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        answer = llm.invoke(prompt)

        # Save chat
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))

# ---------------- Display Chat ----------------
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)