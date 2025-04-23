import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Hugging Face Embedding Model Setup ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# === Paths ===
base_dir = "/Users/suchirmvelpanur/Desktop/Generative AI and its Applications/Project/RAG/kb_docs"
persist_base = "/Users/suchirmvelpanur/Desktop/Generative AI and its Applications/Project/RAG/vector_db"  # Root folder where vector stores will be persisted

# === Text Splitter Configuration ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# === Process each partition ===
for partition in ["arxiv", "pubmed", "blogs"]:
    docs_path = os.path.join(base_dir, partition)
    print(f"\n🔍 Processing {partition} ...")

    # Load all .txt files from this partition
    raw_documents = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_path, filename), encoding='utf-8')
            raw_documents.extend(loader.load())

    # Split into chunks
    documents = text_splitter.split_documents(raw_documents)

    # Vectorize and store using Chroma
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=os.path.join(persist_base, partition)
    )

    # Persist to disk
    vectorstore.persist()
    print(f"✅ Done vectorizing and saving for: {partition}")
