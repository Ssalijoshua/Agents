from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load documents
# Non PDFs
# loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)  # Or PyPDFLoader for PDFs

# For PDF files
loader = PyPDFLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)  # Or PyPDFLoader for PDFs
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
vector_store = FAISS.from_documents(split_docs, embeddings)
vector_store.save_local("faiss_index")  # Save for reuse