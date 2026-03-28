# Import libraries

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Load the PDF
loader = PyPDFLoader("python_tutorial.pdf")
documents = loader.load()

print("PDF Loaded")
print("Pages:", len(documents))

# Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print("Chunks created:", len(docs))

# Step 3: Create local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded")

# Step 4: Store embeddings in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

print("Vector database ready")

# Step 5: Ask a question
query = input("\nAsk a question about the document: ")

# Step 6: Find similar chunks
results = vectorstore.similarity_search(query)

print("\nRelevant sections from document:\n")

for r in results:
    print(r.page_content)
    print("\n----------------------\n")