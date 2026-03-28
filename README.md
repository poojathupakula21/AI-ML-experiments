
# rag_pdf_assistant.py 

# AI PDF RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about a PDF document.

## Features

- Load and parse PDF documents
- Split text into chunks
- Generate embeddings using Sentence Transformers
- Store vectors using FAISS
- Retrieve relevant document sections for user queries

## Tech Stack

- Python
- LangChain
- FAISS
- Sentence Transformers

## Architecture

PDF
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
FAISS Vector DB
↓
User Query
↓
Similarity Search
↓
Relevant Context Returned



# app.py code summary

“I built an AI-powered PDF chatbot using LangChain and Ollama.
The system uses RAG architecture where PDF content is split into chunks, converted into embeddings using HuggingFace models, stored in FAISS, and retrieved based on user queries.
The retrieved context is then passed to a local LLM (llama3 via Ollama) to generate accurate answers.”

Key Concepts You Implemented
✅ RAG (Retrieval-Augmented Generation)
✅ Vector Search
✅ Local LLM (no API cost)
✅ End-to-end AI pipeline
✅ Interactive UI

Architecture in One Line
👉
PDF → Chunking → Embeddings → FAISS → Retrieve → LLM → Answer

# how to run the code

# Terminal 1
ollama serve

# Terminal 2
cd ~/VSCodeProject/ai-pdf-assistant
source .venv/bin/activate
streamlit run app.py


# code explaination

Here's a complete line-by-line explanation of the code in simple terms. I'll break it down step by step, like reading a recipe. Each line or group does something specific to build the PDF chat app. I'll use analogies to make it easy to follow.

**Line 1-6: Getting the tools ready (imports)**
```python
import streamlit as st
```
This brings in a tool called Streamlit, which lets you build a simple website or app without much coding. It's like importing a pre-made kit for making web pages.

```python
from langchain_community.document_loaders import PyPDFLoader
```
This imports a helper that can read PDF files and pull out the text inside. Think of it as a "PDF reader" tool.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```
This imports a tool that breaks long text into smaller pieces. Like cutting a big cake into slices so it's easier to handle.

```python
from langchain_community.vectorstores import FAISS
```
This brings in a smart storage system (FAISS) that organizes text pieces for quick searching. It's like a super-fast filing cabinet for information.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```
This imports a way to turn text into "codes" (numbers) that capture meaning. Like translating words into a secret language that computers understand.

```python
from langchain_ollama import OllamaLLM
```
This imports an AI brain (Llama3 model) that can answer questions. It's like adding a smart assistant who knows a lot.

**Lines 8-10: Setting up the app's look**
```python
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
```
Tells Streamlit what the app's title is and to make the page wide. Like naming your app and choosing a big screen.

```python
st.title("🤖 AI PDF Chat Assistant")
```
Puts a big title on the page. Like writing the app's name at the top.

**Lines 12-19: Preparing memory for the app**
```python
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
```
Checks if the app has a "memory slot" for the smart storage (vectorstore). If not, creates one and sets it to empty. Like making sure there's a place to store your PDF's info.

```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```
Checks for a "memory slot" for past chats. If not, creates an empty list. Like starting a blank notebook for conversations.

**Lines 21-50: Handling PDF uploads**
```python
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")
```
Creates a button on the page where you can drag or select a PDF file. It waits for you to upload one.

```python
if uploaded_file is not None:
```
If you actually uploaded a file (not nothing), then do the next steps.

```python
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
```
Saves the uploaded PDF to your computer as a temporary file called "temp.pdf". Like copying the file to a safe spot.

```python
    st.success("PDF uploaded successfully!")
```
Shows a green success message on the page. Like saying "Upload done!"

```python
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
```
Uses the PDF reader tool to open the file and pull out all the text pages. Like reading the whole book and getting each page's words.

```python
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
```
Sets up the text cutter to break pages into 500-character pieces (with some overlap for context). Then cuts the documents. Like slicing the book into bite-sized chunks.

```python
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
```
Prepares the text-to-code translator using a specific model. Like choosing a translator for the secret language.

```python
    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
```
Takes the text chunks, translates them to codes, and stores them in the smart filing cabinet. Like organizing the book slices into a searchable library.

```python
    st.success("Document processed successfully!")
```
Shows another success message. Like saying "Ready to chat!"

**Lines 52-87: Handling chat questions**
```python
if st.session_state.vectorstore:
```
Only if the smart storage is ready (not empty), show the chat part.

```python
    query = st.chat_input("Ask something about your PDF...")
```
Creates a chat box where you type your question. Like a text input for asking stuff.

```python
    if query:
```
If you typed something (not empty), then process it.

```python
        vectorstore = st.session_state.vectorstore
```
Gets the smart storage from memory.

```python
        results = vectorstore.similarity_search(query, k=3)
```
Searches the storage for the 3 most similar text chunks to your question. Like finding the best matching pages in the book.

```python
        context = "\n".join([doc.page_content for doc in results])
```
Combines the matching text chunks into one big block. Like gluing the relevant pages together.

```python
        llm = OllamaLLM(model="llama3")
```
Gets the AI brain ready.

```python
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
```
Creates a message for the AI: tells it to be helpful, only use the provided text, and answer the question. Like giving instructions to a helper.

```python
        answer = llm.invoke(prompt)
```
Sends the message to the AI and gets back an answer. Like asking the smart assistant and waiting for a reply.

```python
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))
```
Adds your question and the AI's answer to the chat memory. Like writing them in the notebook.

**Lines 89-95: Showing the chat**
```python
for role, message in st.session_state.chat_history:
```
Goes through each item in the chat memory (like flipping through the notebook pages).

```python
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)
```
If it's your message, shows it as "user" chat bubble. If it's AI's, shows it as "assistant" bubble. Like displaying the conversation in a chat app.

That's the whole code! It flows like: Upload PDF → Process it → Ask questions → Get answers → Show chat. If something's still confusing, point to a specific part.